#########################################################################
###  Last updated: April 26, 2022     Jesi Lee                        ###
###                                                                   ###
###                                                                   ###
### This script runs process_compare.py.		              ###
### nist_molecule/ foldeer with reference mass spectra is required.   ###
###                                                                   ###
#########################################################################


## setting variables ##

curdir_path=$PWD
curdir=$(basename $curdir_path)                    
mymol="$(cut -d'_' -f2 <<<"$curdir")"              
mymol_id=$(basename $(dirname $(dirname $PWD)))    
mycidmdjdx=${mymol_id}.cidmd.jdx
mymol_nistfolder=$(find $HOME -type d -iname "nist_${mymol}")



## making links for the reference mass spectra ##

ln -fs ${mymol_nistfolder}/nist_specs/* .
ln -fs ${mymol_nistfolder}/*.JDX .
ln -fs ${mymol_nistfolder}/*.dat .
nistfile=$(ls -1 *_CE*.JDX)
rm process_compare.log
rm simscores.out
rm peaks_predicted_*.out
rm *png



## copying the targeet CIDMD mass spectrum to compare to the reference ##

cp ../results/cidmd.jdx $mycidmdjdx



## running CIDMD_compare.py ##

for i in $nistfile; do python3 CIDMD_compare.py ${mycidmdjdx} ${i} | tee -a process_compare.log ; done 
echo         cid_fname,             nist_fname,      cos_score , dot_score | tee -a simscores.out
echo '* Filename: simscores.out is created.'
cat simscores.out



## making gif and mp4 of headtotail graphs with ffmpeg ##

molid=$(basename $(dirname $(dirname "$PWD")))  
infiles=$(ls ${molid}.h2t.CE*.png)

if [-z "$infiles" ]; then
	echo "infiles do not exist. Quiting." 
	exit 
fi 

tmp=$(mktemp -d)
echo molid
for i in $infiles ; do echo $i ; done
n=1
for i in $infiles ; do cp $i $tmp/$(printf "%03d.png" $n) ; n=$((n+1)) ; done
cwd=$PWD
cd $tmp

ingif=${molid}.h2t.gif
outmp4=${molid}.h2t.mp4
ffmpeg -i %03d.png -vf palettegen palette.png
ffmpeg -framerate .5 -i %03d.png -i palette.png -lavfi paletteuse $ingif
ffmpeg -i $ingif -f mp4 -pix_fmt yuv420p $outmp4

mv $ingif $cwd
mv $outmp4 $cwd
cd $cwd
rm -rf $tmp
echo created $ingif $outmp4

