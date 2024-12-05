import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *

import numpy as np

import plot_format 
plot_format.set_style('ieeetran')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from  transferfunction_analyzer import *

import pdb

class system_response_examples(thesdk):
    def __init__(self):
        self.figpath=self.entitypath+'/Pics'
   
    def single_pole_lp(self):
        """Analysis of a single pole low-pass system 
        """
        tfa=transferfunction_analyzer()
        RC=1
        tfa.poles=[RC]
        tfa.time=np.linspace(0,10/tfa.poles[0],num=100)
        tfa.freq=np.logspace(-3,3,base=10,num=100)*tfa.poles[0]
        dataimp=tfa.time.reshape(-1,1)
        datastep=tfa.time.reshape(-1,1)
        dataamp=tfa.omega.reshape(-1,1)
        dataphase=tfa.omega.reshape(-1,1)
        bodeamp=tfa.omega.reshape(-1,1)
        bodephase=tfa.omega.reshape(-1,1)

        legends=[]
        for rc in [ 1, 2, 10, 1/2, 1/10, ]:
            tfa.poles=[rc]
            # Create plot legends
            legends.append(r'$p_0=%s$' %(rc))
            dataimp=np.r_['-1', dataimp, tfa.imp()]
            datastep=np.r_['-1', datastep, tfa.step()]
            dataamp=np.r_['-1', dataamp, 20*np.log10(np.abs(tfa.tfabs()))]
            dataphase=np.r_['-1', dataphase, tfa.tfphase()]
            bodeamp=np.r_['-1', bodeamp, 20*np.log10(np.abs(tfa.bodeamp()))]
            bodephase=np.r_['-1', bodephase, tfa.bodephase()]

        #Common plot parameters for time domain plot
        xlabel='Time [s]'
        yrange=[-0.3, 1.1]
        ylabel=r'$h\left(t\right)$'
        xtickbase=1
        ytickbase=0.2

        #Impulse response
        self.plot(
                data=dataimp[:,0:2],
                title='Impulse response of a single pole system',
                legends=[legends[0]],
                xlabel=xlabel,
                ylabel=ylabel,
                yrange=yrange,
                xtickbase=xtickbase,
                ytickbase=ytickbase,
                legendloc='lower right',
                figpath=self.figpath
                )

        #Step response
        self.plot(
                data=datastep[:,0:2],
                title='Step response of a single pole system',
                legends=[legends[0]],
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                xtickbase=xtickbase,
                ytickbase=ytickbase,
                legendloc='lower right',
                figpath=self.figpath
                )

        #Step response sweep
        self.plot(
                data=datastep,
                title='Step response sweep of a single pole system',
                legends=legends,
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                xtickbase=xtickbase,
                ytickbase=ytickbase,
                legendloc='lower right',
                figpath=self.figpath
                )

        #Common plot parameters for freqency domain plot
        xlabel='Frequency [Hz]'
        yrange=[-60, 3]
        ylabel=r'$\left|H\left(s\right)\right| [dB]$'

        #Frequency response
        self.plot(
                mode='freq',
                data=dataamp[:,0:2],
                title='Amplitude response of a single pole system',
                legends=legends,
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                ytickbase=20,
                xtickbase=100,
                legendloc='lower right',
                figpath=self.figpath
                )

        #Frequency response sweep
        self.plot(
                mode='freq',
                data=dataamp,
                title='Amplitude response of a single pole system',
                legends=legends,
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                ytickbase=20,
                xtickbase=100,
                legendloc='lower right',
                figpath=self.figpath
                )

        #Phase response
        ylabel=r'$\angle H\left(s\right)  [deg]$'
        yrange=[-130, 10]
        self.plot(
                mode='freq',
                data=dataphase[:,0:2],
                title='Phase response of a single pole system',
                legends=legends,
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                ytickbase=20,
                xtickbase=100,
                legendloc='lower right',
                figpath=self.figpath
                )

        #Frequency response Bode plot
        xlabel='Frequency [Hz]'
        yrange=[-60, 3]
        ylabel=r'$\left|H\left(s\right)\right| [dB]$'
        self.plot(
                mode='freq',
                data=bodeamp[:,0:2],
                title='Amplitude Bode plot of a single pole system',
                legends=legends,
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                ytickbase=20,
                xtickbase=100,
                legendloc='lower right',
                figpath=self.figpath
                )

        #Phase response Bode plot
        ylabel=r'$\angle H\left(s\right)  [deg]$'
        yrange=[-130, 10]
        self.plot(
                mode='freq',
                data=bodephase[:,0:2],
                title='Phase Bode plot a single pole system',
                legends=legends,
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                ytickbase=20,
                xtickbase=100,
                legendloc='lower right',
                figpath=self.figpath
                )
          
        #Frequency response Bode plot
        xlabel='Frequency [Hz]'
        yrange=[-60, 3]
        ylabel=r'$\left|H\left(s\right)\right| [dB]$'
        #data=bodeamp[:,0:2],
        self.plot(
                mode='freq',
                data=np.r_['-1',dataamp[:,0:2],bodeamp[:,1:2]],
                title='Amplitude with Bode plot of a single pole system',
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                ytickbase=20,
                xtickbase=100,
                legendloc='lower right',
                figpath=self.figpath
                )

        #Phase response Bode plot
        ylabel=r'$\angle H\left(s\right)  [deg]$'
        yrange=[-130, 10]
        self.plot(
                mode='freq',
                data=np.r_['-1',dataphase[:,0:2],bodephase[:,1:2]],
                title='Phase with Bode plot a single pole system',
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                ytickbase=20,
                xtickbase=100,
                legendloc='lower right',
                figpath=self.figpath
                )
          
    def single_pole_single_zero(self):
        """Analysis of a single pole low-pass system 
        """
        tfa=transferfunction_analyzer()
        RC=1
        tfa.zeros=[RC]
        tfa.poles=[10*RC]
        tfa.time=np.linspace(0,10/tfa.poles[0],num=100)
        tfa.freq=np.logspace(-3,3,base=10,num=100)/RC
        dataimp=tfa.time.reshape(-1,1)
        datastep=tfa.time.reshape(-1,1)
        dataamp=tfa.omega.reshape(-1,1)
        dataphase=tfa.omega.reshape(-1,1)
        bodeamp=tfa.omega.reshape(-1,1)
        bodephase=tfa.omega.reshape(-1,1)
        zerosbodeamp=tfa.omega.reshape(-1,1)
        zerosbodephase=tfa.omega.reshape(-1,1)

        legends=[]
        print(tfa.impsym)
        print(tfa.stepsym)

        for rc in [  1,10/4, 10/2, 20, 40, ]:
            tfa.zeros=[rc]
            # Create plot legends
            legends.append(r'$p_0=%s,~z_0=%s$' %(tfa.poles[0],rc))
            dataimp=np.r_['-1', dataimp, tfa.imp()]
            datastep=np.r_['-1', datastep, tfa.step()]
            dataamp=np.r_['-1', dataamp, 20*np.log10(np.abs(tfa.tfabs()))]
            dataphase=np.r_['-1', dataphase, tfa.tfphase()]
            bodeamp=np.r_['-1', bodeamp, 20*np.log10(np.abs(tfa.bodeamp()))]
            bodephase=np.r_['-1', bodephase, tfa.bodephase()]
            zerosbodeamp=np.r_['-1', zerosbodeamp, 20*np.log10(np.abs(tfa.zerosbodeamp()))]
            zerosbodephase=np.r_['-1', zerosbodephase, tfa.zerosbodephase()]

        #Common plot parameters for time domain plot
        xlabel='Time [s]'
        yrange=[-100, 1.1]
        ylabel=r'$h\left(t\right)$'
        xtickbase=0.1
        ytickbase=20

        #Impulse response
        #xtickbase=xtickbase,
        self.plot(
                data=dataimp[:,0:2],
                title='Impulse response of a pole-zero system',
                legends=[legends[0]],
                xlabel=xlabel,
                ylabel=ylabel,
                yrange=yrange,
                xtickbase=xtickbase,
                ytickbase=ytickbase,
                legendloc='upper right',
                figpath=self.figpath
                )

        #Step response
        yrange=[0, 10]
        ytickbase=5
        self.plot(
                data=datastep[:,0:2],
                title='Step response of a pole-zero system',
                legends=[legends[0]],
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                xtickbase=xtickbase,
                ytickbase=1,
                legendloc='upper right',
                figpath=self.figpath
                )

        #Step response sweep
        self.plot(
                data=datastep,
                title='Step response sweep of a pole-zero system',
                legends=legends,
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                xtickbase=xtickbase,
                ytickbase=1,
                legendloc='upper right',
                figpath=self.figpath
                )

        #Common plot parameters for freqency domain plot
        xlabel='Frequency [Hz]'
        yrange=[-3, 25]
        ylabel=r'$\left|H\left(s\right)\right| [dB]$'

        #Frequency response
        self.plot(
                mode='freq',
                data=dataamp[:,0:2],
                title='Amplitude response of a pole-zero system',
                legends=legends,
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                ytickbase=5,
                xtickbase=100,
                legendloc='lower right',
                figpath=self.figpath
                )

        #Frequency response zero Bode plot
        xlabel='Frequency [Hz]'
        yrange=[-3, 60]
        ylabel=r'$\left|H\left(s\right)\right| [dB]$'
        self.plot(
                mode='freq',
                data=zerosbodeamp[:,0:2],
                title='Amplitude Bode plot of a single zero system',
                legends=[r'$z_0=1$'],
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                ytickbase=20,
                xtickbase=100,
                legendloc='lower right',
                figpath=self.figpath
                )

        #Phase response zero Bode plot
        xlabel='Frequency [Hz]'
        yrange=[-10, 100]
        ylabel=r'$\left|H\left(s\right)\right| [dB]$'
        self.plot(
                mode='freq',
                data=zerosbodephase[:,0:2],
                title='Phase Bode plot of a single zero system',
                legends=[r'$z_0=1$'],
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                ytickbase=10,
                xtickbase=100,
                legendloc='lower right',
                figpath=self.figpath
                )

        #Frequency response Bode plot
        xlabel='Frequency [Hz]'
        yrange=[-3, 25]
        ylabel=r'$\left|H\left(s\right)\right| [dB]$'
        #data=bodeamp[:,0:2],
        self.plot(
                mode='freq',
                data=np.r_['-1',dataamp[:,0:2],bodeamp[:,1:2]],
                title='Amplitude with Bode plot of a pole-zero system',
                legends=[legends[0], legends[0]],
                yrange=yrange,
                xlabel=xlabel,
                ylabel=ylabel,
                ytickbase=5,
                xtickbase=100,
                legendloc='lower right',
                figpath=self.figpath
                )

    def plot(self,**kwargs):
        """Methods to handle time domain plots
           Assumptions 
           -----------
               Single time axis, multiple values, time is the column 0, rest
               of the comlums are the values to plot.

               Length of the labels is the same as the number of data colums, 
               but if empty, no labels will be added.
           
           Parameters
           ----------
             data : np.array[time,data]
             legends : [ str ] , [ '' ]
             title : str, ''
             yrange, : [ymin, ymax], None
             xtickbase : int, 10
             ytickbase : int, 10 
             legendloc : str, 'upper right'
             figformat : str, 'eps'
             dpi : int,  300
             figpath : 'str, './<title_with_underscores>.<figformat>'

           Returns
           -------
             Handle to a matplotlib figure

        """
        data=kwargs.get('data', None);
        len,datacols=data.shape
        legends=kwargs.get('legends', [ '' for count in range(1,datacols)])
        title=kwargs.get('title',None)
        if not title:
            self.print_log(type='F', msg='Title for the picture must be given.')
        yrange=kwargs.get('yrange',None)
        ylabel=kwargs.get('ylabel','')
        xlabel=kwargs.get('xlabel','')
        ytickbase=kwargs.get('ytickbase', 10)
        xtickbase=kwargs.get('xtickbase', 10)
        legendloc=kwargs.get('legendloc', 'upper right')
        mode=kwargs.get('mode','time')
        
        figformat=kwargs.get('figformat', 'eps')
        dpi=kwargs.get('dpi', 300)
        figpath=kwargs.get('figpath', self.entitypath)
        figfile=kwargs.get('figfile', self.figpath+'/'+title.replace(' ', '_')+'.'+figformat)

        plt.figure()
        h=plt.subplot();
        for col in range(1,datacols):
            if not legends[col-1] == '':
                if mode == 'time':
                    plt.plot(data[:,0], data[:,col], label=legends[col-1])
                elif mode == 'freq':
                    plt.semilogx(data[:,0], data[:,col], label=legends[col-1])

            else:
                if mode == 'time':
                    plt.plot(data[:,0], data[:,col])
                elif mode == 'freq':
                    plt.semilogx(data[:,0], data[:,col])
        plt.ylim(yrange[0],yrange[1]);
        h.yaxis.set_major_locator(ticker.MultipleLocator(base=ytickbase))
        if mode == 'time':
            h.xaxis.set_major_locator(ticker.MultipleLocator(base=xtickbase))
        plt.xlim((data[0,0],data[-1,0]));
        plt.suptitle(title);
        plt.ylabel(ylabel);
        plt.xlabel(xlabel);
        if not legends[0] == '':
                lgd=plt.legend(loc=legendloc);
        plt.grid(True);
        plt.savefig(figfile, format=figformat, dpi=dpi);
        plt.show(block=False);
        return h


if __name__=="__main__":
    import argparse
    # Implement argument parser
    parser = argparse.ArgumentParser(description='Parse selectors')
    parser.add_argument('--show', dest='show', type=bool, nargs='?', const = True,
            default=False,help='Show figures on screen')
    parser.add_argument('--single_pole', dest='single_pole', type=bool, nargs='?', const = True,
            default=False,help='Runb singel pole analysis')
    parser.add_argument('--single_pole_zero', dest='single_pole_zero', type=bool, nargs='?', const = True,
            default=False,help='Run singel pole-zero analysis')
    args=parser.parse_args()
    examples=system_response_examples();
    #if args.single_pole:
    examples.single_pole_lp();
    #if args.single_pole_zero:
    examples.single_pole_single_zero();
    if args.show:
        input()

