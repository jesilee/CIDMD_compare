#!/usr/bin/env python3

#########################################################################
###  Last updated: April 6, 2022     Jesi Lee                         ###
###                                                                   ###
###                                                                   ###
### This script is for Process_compare.sh                             ###
### Need to have the folder Daphnis/ by Yuanyue Li, PhD  for entropy  ###
###                                                                   ###
#########################################################################


import Daphnis.methods
import scipy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import glob



def get_molion_peak(mol_info_infile):

    with open(mol_info_infile) as f:
        mol_name = f.readline().split(' ')[-1].strip()
        mol_nameid = f.readline().split(' ')[-1].strip()
        mol_formula = f.readline().split(' ')[-1].strip()
        mol_weight = f.readline().split(' ')[-1].strip()
        mol_exactmass = f.readline().split(' ')[-1].strip()
    molion_peak = mol_weight 
    return molion_peak



def calc_entropy_simscore(cidmd_jdx_in, nist_jdx_in):
    
    cidmd_array = np.genfromtxt(cidmd_jdx_in,delimiter=' ' , dtype=float)
    nist_array = np.genfromtxt(nist_jdx_in, delimiter=' ' , dtype=float)
    
    entropy_simscore = 1 - Daphnis.methods.distance(cidmd_array, nist_array, method="weighted_entropy", ms2_da=0.05)
    return entropy_simscore 



def read_ms_spec(infilename, normalize_to=0):

    spec_pd = pd.read_csv(infilename, comment='#', header=None, names=("mz", "intensity"), dtype=({"mz": np.float64, "intensity": np.float64}),skipinitialspace=True, delim_whitespace=True)
    spec_mz_np = spec_pd.mz.to_numpy()
    spec_intensity_np = spec_pd.intensity.to_numpy()
    num_mzs = len(spec_mz_np)
    spec_mz = spec_mz_np.reshape(num_mzs,1)
    spec_intensity = spec_intensity_np.reshape(num_mzs,1)
    if normalize_to != 0:
        spec_intensity /= spec_intensity.max()
        spec_intensity *= normalize_to
    return spec_pd, spec_mz, spec_intensity



def gen_sim_vec(cidmd_mz, cidmd_intensity, nist_mz, nist_intensity):

    mz_max = 0
    if cidmd_mz.max() > nist_mz.max():
        mz_max = cidmd_mz.max()
    else:
        mz_max = nist_mz.max()
    cidmd_mz = cidmd_mz.astype(np.int32)
    nist_mz = nist_mz.astype(np.int32)
    vec_len = int(mz_max) + 20
    cidmd_vec = np.zeros([vec_len,])
    nist_vec = np.zeros([vec_len,])
    for i in range( len(cidmd_mz)):
        cidmd_vec[cidmd_mz[i]] = cidmd_intensity[i]
    for i in range( len(nist_mz)):
        nist_vec[nist_mz[i]] = nist_intensity[i]
    return cidmd_vec, nist_vec


def cos_sim(a, b):
    ''' calculate cosine similarity of two vetcors
    '''
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    res = dot_product / (norm_a * norm_b)
    return res*1000.


def dot_sim(x,y):
    ''' calculate dot product of two mass spectrums
    '''
    m=0.6
    n=3
    #must use different parameters!!!otherwise you will destory the initial array!!
    a = np.copy(x)
    b = np.copy(y)
    for i in range(0, len(a)-1):
        a[i] = (i**n)*(a[i]**m)
    for i in range(0, len(b)-1):
        b[i] = i**n*(b[i]**m)
    dot= np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))*1000
    return dot



def plot_headtotail(cidmd_jdx_in, nist_jdx_in, saveto=None):
    qcjdx, cidmd_mz, cidmd_intensity = read_ms_spec(cidmd_jdx_in, normalize_to=100) #top aka target = cidmd 
    nijdx, nist_mz, nist_intensity = read_ms_spec(nist_jdx_in, normalize_to=-100)  #bottom aka ref.= nist

    fig = plt.figure(figsize=(20,14), dpi=350)
    matplotlib.rc("font", **{ "size": 28 })
    ax = fig.add_subplot( 111)

    qc_spec = ax.bar( qcjdx.mz, qcjdx.intensity, width=0.8, label='Theoretical spectrum', color='magenta')
    ni_spec = ax.bar( nijdx.mz, nijdx.intensity, width=0.8, label='Experimental spectrum', color='blue')

    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_ylabel('Intensity')
    ax.set_xlabel('m/z')
    ax.xaxis.set_label_coords(0.96, 0.48)
    ax.legend(fontsize=20,loc='best')

    cid_id = cidmd_jdx_in.split('.')[0]    
    nist_id = nist_jdx_in.split('_')[-1].replace('.JDX','')    

    ax.set_title("Molecule " + cid_id+".h2t."+nist_id, fontsize=14)

    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=20)

    label_peaks_headtotail(ax, qc_spec)
    label_peaks_headtotail(ax, ni_spec)

    ax.set_yticks(list(range(-100,101,25))) 
    ticks = ax.get_yticks() 
    ax.set_yticklabels([int(abs(tick)) for tick in ticks])
    
    trim_xaxis = True
    if trim_xaxis:
        min_mz = 0.
        # defined as the mz with the maximum intensity
        max_mz = max( qcjdx.mz.max(), nijdx.mz.max() )
        ax.set_xlim(min_mz, 1.1* max_mz)
    fig.tight_layout()
    if saveto:
        fig.savefig(saveto)

    return fig



def peaks_headtotail( x, y, separation=1., ratio=1.01, threshold=0.0):
    """
    separation means the peak should be separated in mz by +- this number 
    ratio means the peak should be this many times greater than the other points
    threshhold means the intensity should be at least this number to be a peak
    """
    assert separation > 0 and ratio > 0
    x = np.asarray(x)
    y = np.asarray(y)

    # sort them to make sure we always prefer the largest peaks first
    srt = np.abs(y).argsort()[::-1]

    xp = []
    yp = []

    for i in range( len(x)):
        i = srt[i]
        mask = np.abs(x - x[i])
        mask = mask < separation
        if not mask.any():
            #print("skipping due to empty mask")
            continue
        group = y[mask]

        # quick way to just get the 2 largest values
        if len(group) > 1:
            max_indices = np.argpartition(np.abs(group), len(group) - 2)[-2:]
            xval = x[mask][ max_indices[1]]
            yval = group[ max_indices[1]]
            second_max = group[ max_indices[0]]
        else:
            idx = group.argmax()
            xval = x[mask][ idx]
            yval = group[ idx]
            second_max = 0
        if second_max == 0:
            ratio_i = ratio * 2
        else:
            ratio_i = yval / second_max
        uniq = [ abs(xpi - xval) > separation for xpi in xp]
        xp.append( xval)
        yp.append( yval)
    return xp, yp



def label_peaks_headtotail(ax, rects, peaks_only=True):

    dat = [[rect.get_x() + rect.get_width()/2., rect.get_height()] for rect in rects]
    if peaks_only:
        x = [v[0] for v in dat]
        y = [v[1] for v in dat]
        xp, yp = peaks_headtotail( x, y)
    else:
        xp = [v[0] for v in dat if v[1] > 0.0]
        yp = [v[1] for v in dat if v[1] > 0.0]
    for mz, intensity in zip( xp, yp):
        ax.annotate('%g' % mz, (mz, np.sign(intensity)*7 + intensity), ha='center', va='center', fontsize=18)





def plot_headtotail_slide(cidmd_jdx_in, nist_jdx_in, saveto=None):
    qcjdx, cidmd_mz, cidmd_intensity = read_ms_spec(cidmd_jdx_in, normalize_to=100) #top aka target = cidmd 
    nijdx, nist_mz, nist_intensity = read_ms_spec(nist_jdx_in, normalize_to=-100)  #bottom aka ref.= nist

    fig = plt.figure(figsize=(20,14), dpi=350)
    matplotlib.rc("font", **{ "size": 48 })
    ax = fig.add_subplot( 111)

    qc_spec = ax.bar( qcjdx.mz, qcjdx.intensity, width=0.8, label='Theoretical spectrum', color='magenta')
    ni_spec = ax.bar( nijdx.mz, nijdx.intensity, width=0.8, label='Experimental spectrum', color='blue') 
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_ylabel('Intensity')
    ax.set_xlabel('m/z')
    ax.xaxis.set_label_coords(0.96, 0.48)
    ax.legend(fontsize=40,loc='best')
    
    cid_id = cidmd_jdx_in.split('.')[0]    
    nist_id = nist_jdx_in.split('_')[-1].replace('.JDX','')    

    ax.set_title("Molecule " + cid_id+".h2t."+nist_id, fontsize=14)
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=30)

    label_peaks_headtotail_slide(ax, qc_spec)
    label_peaks_headtotail_slide(ax, ni_spec)

    ticks = ax.get_yticks() 
    ax.set_yticklabels([int(abs(tick)) for tick in ticks])
    

    trim_xaxis = True
    if trim_xaxis:
        min_mz = 0.
        max_mz = max( qcjdx.mz.max(), nijdx.mz.max() )
        ax.set_xlim(min_mz, 1.1* max_mz)
    fig.tight_layout()
    if saveto:
        fig.savefig(saveto)

    return fig




def peaks_headtotail_slide( x, y, separation=3., ratio=1.01, threshold=0.0):
    """
    separation means the peak should be separated in mz by +- this number 
    ratio means the peak should be this many times greater than the other points
    threshhold means the intensity should be at least this number to be a peak
    """
    assert separation > 0 and ratio > 0
    x = np.asarray(x)
    y = np.asarray(y)

    # sort them to make sure we always prefer the largest peaks first
    srt = np.abs(y).argsort()[::-1]

    xp = []
    yp = []

    for i in range( len(x)):
        i = srt[i]
        mask = np.abs(x - x[i])
        mask = mask < separation
        if not mask.any():
            #print("skipping due to empty mask")
            continue
        group = y[mask]

        if len(group) > 1:
            max_indices = np.argpartition(np.abs(group), len(group) - 2)[-2:]
            xval = x[mask][ max_indices[1]]
            yval = group[ max_indices[1]]
            second_max = group[ max_indices[0]]
        else:
            idx = group.argmax()
            xval = x[mask][ idx]
            yval = group[ idx]
            second_max = 0
        
        if i > 0:
            if np.abs( y[ i]) < np.abs( y[ i-1]):
                continue
        if i < len(y) - 1:
            if np.abs( y[ i]) < np.abs( y[ i+1]):
                continue
        if second_max == 0:
            ratio_i = ratio * 2
        else:
            ratio_i = yval / second_max

        if ratio_i > ratio and np.abs(yval) > threshold:
            uniq =  [ abs(xpi - xval) > separation for xpi in xp]
            if all(uniq):
                xp.append( xval) 
                yp.append( yval)
        uniq = [ abs(xpi - xval) > separation for xpi in xp]
        xp.append( xval)
        yp.append( yval)
    return xp, yp




def label_peaks_headtotail_slide(ax, rects, peaks_only=True):
    """
    Attach a text label above each bar displaying its intensity
    """

    dat = [[rect.get_x() + rect.get_width()/2., rect.get_height()] for rect in rects]
    if peaks_only:
        x = [v[0] for v in dat]
        y = [v[1] for v in dat]
        xp, yp = peaks_headtotail_slide( x, y)
    else:
        xp = [v[0] for v in dat if v[1] > 0.0]
        yp = [v[1] for v in dat if v[1] > 0.0]
    for mz, intensity in zip( xp, yp):
        ax.annotate('%g' % mz, (mz, np.sign(intensity)*7 + intensity), ha='center', va='center', fontsize=30)





def generate_headtotail(cidmd_jdx_in, nist_jdx_in):
    
    cid_id = cidmd_jdx_in.split('.')[0]    
    nist_id = nist_jdx_in.split('_')[-1].replace('.JDX','')    
    h2t_outgraph = cid_id+".h2t."+nist_id+".png"
    fig = plot_headtotail(cidmd_jdx_in, nist_jdx_in, saveto= cid_id+".h2t."+nist_id+".png")   # must read it in here due to the range(100, -100) 
    print('\n* Filename:', h2t_outgraph, '     (raw ver.  ) is created.\n')
    return fig




def generate_headtotail_slide(cidmd_jdx_in, nist_jdx_in):
    cid_id = cidmd_jdx_in.split('.')[0]    
    nist_id = nist_jdx_in.split('_')[-1].replace('.JDX','')    
    fig = plot_headtotail_slide(cidmd_jdx_in, nist_jdx_in, saveto= cid_id+".h2t_slide."+nist_id+".png")    
    h2t_outgraph = cid_id+".h2t_slide."+nist_id+".png"
    print('\n* Filename:', h2t_outgraph, '(slide ver.) is created.\n')
    return fig



def get_peaks_tohave_minmax(peaks_tohave, peaks_tohave_thresh ):
    peaks_tohave_min=[]
    peaks_tohave_max=[]
    for i in peaks_tohave:
        check_peak = float(i)
        check_peak_min = i - peaks_tohave_thresh
        check_peak_max = i + peaks_tohave_thresh
        peaks_tohave_min.append(check_peak_min)
        peaks_tohave_max.append(check_peak_max)
    return peaks_tohave_min, peaks_tohave_max



def get_predicted_peaks(peaks_tohave, peaks_tohave_min, peaks_tohave_max, cidmd_jdx  ):
    predicted = []
    for j in range(len(peaks_tohave)):
        #print("checking ", peaks_tohave[j])
        for i in range(len(cidmd_jdx)):
            if peaks_tohave_min[j] < cidmd_jdx[i,0] < peaks_tohave_max[j]:
                predicted.append(cidmd_jdx[i,0])
    return predicted



def check_peaks(cidmd_spec_pd, peaks_tohave, peak_min, peak_max ):
    cidmd_spec_np = cidmd_spec_pd.to_numpy()
    cidmd_spec_np = np.round(cidmd_spec_np, 5)
    keep_cidmd_peaks=[]
    keep_nist_peaks=[]
    for i in range(len(cidmd_spec_np)) :
        for j in range(len(peaks_tohave)):
            if peak_min[j] <  cidmd_spec_np[i,0] < peak_max[j]:
                keep_cidmd_peaks.append(cidmd_spec_np[i])
                keep_nist_peaks.append(peaks_tohave[j])
    np.set_printoptions(suppress=True)
    cidmd_peaks = np.round(keep_cidmd_peaks, 5)
    nist_peaks = np.round(keep_nist_peaks, 5)
    missing_peaks = np.setdiff1d(peaks_tohave, keep_nist_peaks)
    return cidmd_peaks, nist_peaks, missing_peaks



def save_peaks(peaks_outfilename, peaks_tohave, missing_peaks, keep_nist_peaks, keep_cidmd_peaks):
    # this func can't stand alone. must be part of generate_predicted_peaks_tohave()
    f = open(peaks_outfilename,'w')
    f.write('# nist_peaks_tohave    : mz = %s \n' % peaks_tohave )
    f.write('# cidmd_peaks_missing  : mz = %s \n' % missing_peaks )
    f.write('#  nist_mz ,  cidmd_mz , cidmd_intensity\n')
    for nist_peak, (mz, intensity) in zip(keep_nist_peaks, keep_cidmd_peaks):
        f.write(f"{nist_peak:10.2f} {mz:10.2f} {intensity*1000:10.2f}\n")
    f.close()



def save_simscores(simscores_outfilename, cidmd_jdx_in, nist_jdx_in, cos_score, dot_score, ent_score ):
    f= open(simscores_outfilename, 'a')
    f.write(f"{cidmd_jdx_in:25s} {nist_jdx_in:20s} {cos_score:10.4f} {dot_score:10.4f} {ent_score:10.4}\n")
    f.close()



def generate_predicted_peaks_tohave( peaks_infilename, peak_thresh, peaks_outfilename ):
    peaks_tohave = np.genfromtxt(peaks_infilename, delimiter=',', dtype=float)
    peak_min, peak_max = get_peaks_tohave_minmax(peaks_tohave, peak_thresh)
    keep_cidmd_peaks, keep_nist_peaks, keep_missing_peaks = check_peaks(cidmd_spec, peaks_tohave, peak_min, peak_max)
    save_peaks(peaks_outfilename, peaks_tohave, keep_missing_peaks, keep_nist_peaks, keep_cidmd_peaks)



def main():

    cidmd_jdx_in = sys.argv[1]  
    nist_jdx_in = sys.argv[2]   
    mol_info_infile = '../../../../mol_info.in'
    molion_peak = get_molion_peak(mol_info_infile)
    
    # step 1. pre-process: reading ms spec of cidmd and nist. #
    cidmd_spec, cidmd_mz, cidmd_intensity = read_ms_spec(cidmd_jdx_in, normalize_to=1)
    nist_spec, nist_mz, nist_intensity = read_ms_spec(nist_jdx_in, normalize_to=1)
    
    
    # step 2. calculating simscores not including molecular ion peak. # 
    cidmd_vec, nist_vec = gen_sim_vec(cidmd_mz, cidmd_intensity, nist_mz, nist_intensity) 
    cidmd_vec[molion_peak] = 0.
    nist_vec[molion_peak] = 0.
    cos_score = cos_sim(cidmd_vec, nist_vec)
    dot_score = dot_sim(cidmd_vec, nist_vec)
    ent_score = calc_entropy_simscore(cidmd_jdx_in, nist_jdx_in)
    save_simscores('simscores.out',cidmd_jdx_in, nist_jdx_in, cos_score, dot_score, ent_score )
   

    # step 3. making h2t graphs. #
    generate_headtotail(cidmd_jdx_in, nist_jdx_in)
    generate_headtotail_slide(cidmd_jdx_in, nist_jdx_in)
   

    # step 4. checking if cidmd_jdx contains peaks_tohave.dat. #
    peak_thresh=0.3
    peaks_tohave_infiles = np.array(sorted(glob.glob('peaks_tohave_*.dat')))
    
    for i in range(len(peaks_tohave_infiles)):
        ms_type=peaks_tohave_infiles[i].split('.')[0].split('_')[-1]
        peaks_infilename = peaks_tohave_infiles[i]
        peaks_outfilename = 'peaks_predicted_'+ms_type+'.out'
        generate_predicted_peaks_tohave( peaks_infilename, peak_thresh, peaks_outfilename )
    print(' Processing CIDMD_compare done. please check the outputs')
    

if __name__=="__main__":
    main()


