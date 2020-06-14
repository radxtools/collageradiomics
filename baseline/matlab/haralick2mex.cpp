#include "mex.h"
#include <math.h>
#include <cmath>
#include <cstdlib>

#ifndef min
#define min(X, Y)  (X < Y ? X : Y)
#endif
#ifndef max
#define max(X, Y)  (X < Y ? Y : X)
#endif
#ifndef isnan 
#define isnan(x) ((x)!=(x)) 
#endif
/* haralick2mex -- Haralick for 2D images. Syntax:
 * haralickims = haralick2mex(double image, double graylevels, double window_size, double dist, double background [optional]) */

 /* To Compile:
 *-WINDOWS (64-bit, Visual Studio)
 * mex haralick2mex.cpp
 */
 
//typedef unsigned int uint;
//typedef unsigned short uint16;

inline bool greater (int i,int j) { return (j<i); }

inline double logb(double x, double b)
{
  return log(x)/log(b);
}

// bool utSetInterruptEnabled(bool);
// bool utIsInterruptPending(void);
void graycomtx(const double *image, double *comtx, int ws, int dist, 
        int graylevels, const int background, int rows, int cols, int row, int col) {
    
    int i, j, k, l, centerind, pixind, center_value, hws;
    int d_start_row, d_start_col, d_end_row, d_end_col;
    int block_start_row, block_start_col, block_end_row, block_end_col;
    
    for (i = 0; i < graylevels*graylevels; i++)
        comtx[i] = 0.0;
    
    hws=(int) floor((float) ws/2);
    block_start_row = max(0, row-hws);
    block_start_col = max(0, col-hws);
    block_end_row = min(rows-1, row+hws);
    block_end_col = min(cols-1, col+hws);
    
    //mexPrintf("\nstart_row, start_col, end_row, end_col = %i, %i, %i, %i \n", block_start_row, block_start_col, block_end_row, block_end_col);
    
    for (j = block_start_col; j < block_end_col; j++)  {
        for (i = block_start_row; i < block_end_row; i++) {
            centerind=i+j*rows;
            center_value = (int) image[centerind];
			if (isnan(image[centerind]) || (image[centerind] == background))
                continue;
			
			center_value = (int) image[centerind];

	        d_start_row = max((int) 0, i-dist);
            d_start_col = max((int) 0, j-dist);
            d_end_row = min((int) rows-1, i+dist);
            d_end_col = min((int) cols-1, j+dist);
            for (l = d_start_col; l <= d_end_col; l++) {
                for (k = d_start_row; k <= d_end_row; k++) {
                    pixind=k+l*rows;
                    //if (row == 0 && col == 0) mexPrintf("\nl, k = %i, %i\n", l, k);
                    if (!isnan(image[pixind]) && (image[pixind]!=background)) {
						//if ((dist==0) || (pixind!=centerind)) //either dist=0 or exclude dist=0
                        comtx[center_value + (int) (image[pixind]+0.5)*graylevels] += 1;
					}
                }
			}
        }
	}

	//testing purposes
  /*
  int sum = 0;
  for (i = 0; i < graylevels*graylevels; i++) {
    sum += comtx[i];
  }
  mexPrintf("\nx, y, Sum = %i, %i, %i", col, row, sum);
  
  for (i = 0; i < graylevels*graylevels; i++) {
       if ((int) comtx[i] != 0)
           mexPrintf("\nWith window centered at: [%i][%i]: comtx[%i]=%i",row,col,i,(int) comtx[i]);
    }
   mexPrintf(" (all else zeros)\n");   
  */
  
}
		
void haralick2(double *image, double *haralicks, int ws, int dist, int graylevels, int background, int rows, int cols, int nharalicks) {
    
    int i, j, k, ii, jj, nbins, nzeros, nnonzeros, somepct, tenpct, pynzs, pxnzs;
    int *hi, *hj, *himhj, *hiphj;
    double *comtx, *p, *pnz, *nzcomtx, *px, *py, *pxplusy, *pxminusy;
    double entropyval, energyval, inertiaval, idmval, 
            correlationval, info1val, info2val, H1, H2,
            sigma_x, sigma_y, mu_x, mu_y, h_x, h_y, h_max,
            saval, svval, seval, daval, dvval, deval, cosum;
    
    nbins=graylevels*graylevels;
    somepct = (int) floor(.025*rows*cols-1);
    tenpct = (int) floor(.1*rows*cols-1);
    
    comtx = (double *) mxMalloc(nbins*sizeof(double));
    nzcomtx = (double *) mxMalloc(nbins*sizeof(double));
    
    p = (double *) mxMalloc(nbins*sizeof(double));
    pnz = (double *) mxMalloc(nbins*sizeof(double));
    px = (double *) mxMalloc(graylevels*sizeof(double));
    py = (double *) mxMalloc(graylevels*sizeof(double));
    pxplusy = (double *) mxMalloc(2*graylevels*sizeof(double));
    pxminusy = (double *) mxMalloc(graylevels*sizeof(double));
    
    hi = (int *) mxMalloc(nbins*sizeof(int));
    hj = (int *) mxMalloc(nbins*sizeof(int));
    himhj = (int *) mxMalloc(nbins*sizeof(int));
    hiphj = (int *) mxMalloc(nbins*sizeof(int));
    
    for(j = 0; j < cols; j++)
        for(i = 0; i < rows; i++)
            if(image[i+j*rows] >= graylevels && image[i+j*rows]!=background)
                mexErrMsgTxt("Graylevels of image fall outside acceptable range.");
    
    for (j=0; j<cols; j++) {
        for (i=0; i<rows; i++) {
            if (image[i+j*rows]!=background) {
            /* Get co-occurrence matrix */
            graycomtx(image, comtx, ws, dist, graylevels, background, rows, cols, i, j);
            
            /* Initialize feature values */
            entropyval=0; energyval=0; inertiaval=0; idmval=0;
            correlationval=0; info1val=0; info2val=0;
            saval=0; svval=0; seval=0; daval=0; dvval=0; deval=0;
            H1=0; H2=0; h_x=0; h_y=0; h_max=0; mu_x=0; mu_y=0; sigma_x=0; sigma_y=0;
            cosum=0;
            
            /* Non-zero elements & locations in comtx and distribution */
            //nzeros=std::count(comtx,comtx+nbins,0);
            //nnonzeros=nbins-nzeros;
            for (k=0; k<nbins; k++) cosum+=comtx[k];
            if (cosum<2) continue;
            for (k=0, ii=0; k<nbins; k++) {
                if (comtx[k]>0) {
                    p[k]=comtx[k]/cosum;
                    pnz[ii]=p[k];
                    nzcomtx[ii]=comtx[k];
                    hi[ii]=k % graylevels;
                    hj[ii]=(int) floor((float) k/(float) graylevels);
                    himhj[ii]=hi[ii]-hj[ii];
                    hiphj[ii]=hi[ii]+hj[ii];
                    ii++;
                } else {
                    p[k]=0;
                }
            }
            nnonzeros=ii; nzeros=nbins-nnonzeros;
            
            /* Entropy, Energy, Inertial, Inv. Diff. Moment */
            for (k=0; k<nnonzeros; k++) {
                //pnz[k]=nzcomtx[k]/nbins;
                entropyval-=pnz[k]*logb(pnz[k],2.0);
                energyval+=pnz[k]*pnz[k];
                inertiaval+=himhj[k]*himhj[k]*pnz[k];
                idmval+=pnz[k]/(1.0+himhj[k]*himhj[k]);
            }
            
            /* Marginal distributions */
            for (ii=0; ii<graylevels; ii++) { px[ii]=0; py[ii]=0; }
            for (k=0, ii=0; ii<graylevels; ii++)
                for (jj=0; jj<graylevels; jj++, k++) {
                    py[ii]+=p[k];
                    px[jj]+=p[k];
                }
            /*
            for (ii=0, pynzs=0, pxnzs=0; ii<graylevels; ii++) {
                pynzs+=py[ii]>0;
                pxnzs+=px[ii]>0;
            }
            if (pynzs<2 || pxnzs<2) continue;
             */
            
            /* Correlation */
            for (ii=0; ii<graylevels; ii++) {
                h_x-=(px[ii]>0 ? px[ii]*logb(px[ii],2.0) : 0);
                h_y-=(py[ii]>0 ? py[ii]*logb(py[ii],2.0) : 0);
                mu_x+=ii*px[ii];
                mu_y+=ii*py[ii];
            }
            
            for (ii=0; ii<graylevels; ii++) {
                sigma_x+=pow(ii-mu_x,2) * px[ii];
                sigma_y+=pow(ii-mu_y,2) * py[ii];
            }
            
            if (sigma_x>(1e-4) && sigma_y>(1e-4)) {
                for (k=0; k<nnonzeros; k++)
                    correlationval+=(hi[k]-mu_x)*(hj[k]-mu_y)*pnz[k];
                correlationval/=sqrt(sigma_x*sigma_y);
            } else correlationval=0;
            
            /* Information measures of correlation */
             for (k=0, ii=0; ii<graylevels; ii++)
                for (jj=0; jj<graylevels; jj++, k++) {
                    H1-=(p[k]>0 && px[jj]>0 && py[ii]>0 ? p[k]*logb(px[jj]*py[ii],2.0) : 0);
                    H2-=(px[jj]>0 && py[ii]>0 ? px[jj]*py[ii]*logb(px[jj]*py[ii],2.0) : 0);
                }
            h_max=max(h_x,h_y);
            info1val=(h_max!=0 ? (entropyval-H1)/h_max : 0);
            info2val=sqrt(abs( (1-exp(-2*(H2-entropyval)))  ) );
            
            /* Sum average, variance and entropy */
            for (k=0; k<(2*graylevels); k++)
                pxplusy[k]=0;
            for (k=0; k<nnonzeros; k++)
                pxplusy[hiphj[k]]+=pnz[k];
            
            for (k=0; k<(2*graylevels); k++) {
                saval+=k*pxplusy[k];
                seval-=(pxplusy[k]>0 ? pxplusy[k]*logb(pxplusy[k],2.0) : 0);
            }
            for (k=0; k<(2*graylevels); k++)
                svval+=pow(k-saval,2) * pxplusy[k];
                
            /* Difference average, variance and entropy */
            for (k=0; k<graylevels; k++)
                pxminusy[k]=0;
            for (k=0; k<nnonzeros; k++)
                pxminusy[abs(himhj[k])]+=pnz[k];
            
            for (k=0; k<graylevels; k++) {
                daval+=k*pxminusy[k];
                deval-=(pxminusy[k]>0 ? pxminusy[k]*logb(pxminusy[k],2.0) : 0);
            }
            for (k=0; k<graylevels; k++)
                dvval+=pow(k-daval,2) * pxminusy[k];
            
            /* Work on unsorted comtx */
            /*
            for (k=0; k<nbins; k++) {
                p[k]=comtx[k]/nbins;
                entropyval-=(p[k]>0 ? p[k]*log(p[k]) : 0);
                energyval+=p[k]*p[k];
            }
             */
            
            /* Sorted comtx */
            /*
            std::sort(comtx,comtx+nbins,greater);
            zeroloc=std::find(comtx,comtx+nbins,0);
            
            for (k=0; k<zeroloc; k++) {
                p[k]=comtx[k]/nbins;
                entropyval-=p[k]*log(p[k]);
                energyval+=p[k]*p[k];
            }
             */
            
            /* Put feature values in output volume */
            
            // these should correlate to mahotas python library and original paper #'s
            haralicks[i+j*rows+0*rows*cols]=energyval;
            haralicks[i+j*rows+1*rows*cols]=daval;
            haralicks[i+j*rows+2*rows*cols]=correlationval;
            haralicks[i+j*rows+3*rows*cols]=inertiaval;
            haralicks[i+j*rows+4*rows*cols]=idmval;
            haralicks[i+j*rows+5*rows*cols]=saval;
            haralicks[i+j*rows+6*rows*cols]=svval;
            haralicks[i+j*rows+7*rows*cols]=seval;
            haralicks[i+j*rows+8*rows*cols]=entropyval;
            haralicks[i+j*rows+9*rows*cols]=dvval;
            haralicks[i+j*rows+10*rows*cols]=deval;
            haralicks[i+j*rows+11*rows*cols]=info1val;
            haralicks[i+j*rows+12*rows*cols]=info2val;
            
            } else {
                for (k=0; k<nharalicks; k++) haralicks[i+j*rows+k*rows*cols]=0;
            }
            /**
            if (((i+j*rows+1) % somepct)==0) {
                mexPrintf("."); mexEvalString("drawnow");
            } else if (((i+j*rows) % tenpct)==0) {
                mexPrintf("%d%%",(int) ceil((float) 100*(i+j*rows)/(rows*cols-1))); mexEvalString("drawnow");
//                 if (utIsInterruptPending()) return;
            }
			*/
			
			
        }
    }
    //mexPrintf("\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    double *haralicks, *image;
    int dist, rows, cols;
    int graylevels, background, ws;
    mwSize dims[3];
    int nharalicks=13;
    //unsigned short int *X;
    mexPrintf("\nharalick2mex called.\n");
	
    if(nrhs > 5 || nrhs < 4)
        mexErrMsgTxt("haralick2mex(image,graylevels,ws,dist,[background])");
    
    if(!mxIsDouble(prhs[0]))
        mexErrMsgTxt("Input image must be DOUBLE.");
    
    image = mxGetPr(prhs[0]);
    rows = (int) mxGetM(prhs[0]);
    cols = (int) mxGetN(prhs[0]);
    graylevels=(int) mxGetScalar(prhs[1]);
    ws = (int) mxGetScalar(prhs[2]);
    dist = (int) mxGetScalar(prhs[3]);
    if (nrhs==4)
        background = -1;
    else
        background = (int) mxGetScalar(prhs[4]);
    
    if(graylevels < 0 || graylevels > 65535)
        mexErrMsgTxt("GRAYLEVELS must be between 0 and 2^16-1.");
    
    dims[0] = rows; dims[1] = cols; dims[2] = nharalicks;
    plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    haralicks = mxGetPr(plhs[0]);
    
    haralick2(image,haralicks,ws,dist,graylevels,background,rows,cols,nharalicks);
}
