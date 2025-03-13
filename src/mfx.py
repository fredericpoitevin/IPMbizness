from psana import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

class IPMReader:
    def __init__(self, exp, run,
            components=['mirror_pitch', 'dg1', 'dg2'],
            mirror_pitch_pv='MR1L4:HOMS:MMS:PITCH',
            dg1_ipm_pv='MFX:DG1:W8:01',dg1_ipm_version=3,
            dg2_ipm_pv='MFX-DG2-BMMON',dg2_ipm_version=1):
        """
        Detectors - to get full list, do DetName('detectors'), DetNames('epics') or DetNames('all'):
        - DG1 (DAQ recorded): MFX-DG1-BMMON
        - DG2 (DAQ recorded): MFX-DG2-BMMON
        - Mirror (EPICS): MR1L4:HOMS:MMS:PITCH
        """
        self.exp = exp
        self.run = run
        self.beamline = {}
        self.beamline['components'] = components
        self.beamline['mirror_pitch'] = {'pv':mirror_pitch_pv}
        self.beamline['dg1'] = {'pv':dg1_ipm_pv, 'version':dg1_ipm_version}
        self.beamline['dg2'] = {'pv':dg2_ipm_pv, 'version':dg2_ipm_version}
        self.setup()

    def setup(self):
        self.ds = DataSource(f"exp={self.exp}:run={self.run}:smd")
        for dg in ['dg1', 'dg2']:
            if dg in self.beamline['components']:
                if self.beamline[dg]['version'] == 3:
                    for i in range(8):
                        self.beamline[dg][f"det{i}"] = Detector(f"{self.beamline[dg]['pv']}:CH{i}:ArrayData")
                else:
                    self.beamline[dg]['det'] = Detector(self.beamline[dg]['pv'])
        if 'mirror_pitch' in self.beamline['components']:
            self.beamline['mirror_pitch']['det'] = Detector(self.beamline['mirror_pitch']['pv'])
        print(self.beamline)

    def init_run(self):
        #self.ds = DataSource(f"exp={self.exp}:run={self.run}:smd")
        self.num_events = {}
        self.num_events['total'] = 0
        self.num_events['damaged'] = {}
        self.num_events['damaged']['total'] = 0
        self.beamline['time_s'] = []
        self.beamline['event_id'] = []
        for dg in ['dg1', 'dg2']:
            if dg in self.beamline['components']:
                self.num_events['damaged'][dg] = 0
                self.beamline[dg]['det_event'] = []
                self.beamline[dg]['Xpos'] = []
                self.beamline[dg]['TotInt'] = []
        if 'mirror_pitch' in self.beamline['components']:
            self.beamline['mirror_pitch']['RBV'] = []

    def skip_event(self, evt, nevent_start=0, nevent_end=-1):
        self.num_events['total'] += 1
        skip_code =  0
        for dg in ['dg1', 'dg2']:
            if dg in self.beamline['components']:
                self.beamline[dg]['det_event'] = self.get_det_event(evt, dg)
                if self.beamline[dg]['det_event'] is None:
                    self.num_events['damaged'][dg] += 1
                    skip_code =  1
        if skip_code == 1:
            self.num_events['damaged']['total'] += 1
        if self.num_events['total'] < nevent_start:
            skip_code = 1
        if nevent_end > 0:
            if self.num_events['total'] > nevent_end:
                skip_code = 2
        return skip_code

    def get_det_event(self, evt, dg="dg1"):
        if self.beamline[dg]['version'] == 1:
            return self.beamline[dg]['det'].get(evt)
        else:
            det_event_list = []
            for i in range(8):
                det_event_list.append(self.beamline[dg][f"det{i}"]())
            return np.array(det_event_list)

    def get_event_time(self, evt):
        evt_time = evt.get(EventId).time()
        return (evt_time[0]*1e9 + evt_time[1])*1e-9

    def process_wave8v3(self,det_evt):
        IPM_intensities = np.zeros(8)
        for i in range(8):
            counter=0
            avg=0
            for j in range(25,75):
                avg = (avg*counter + (det_evt[i][j]))/(counter+1)
                counter += 1
            maximum = 0 
            peakT = 0
            for j in range(100,115):
                value = abs(avg-det_evt[i][j])
                if(value>maximum):
                    maximum = value
            IPM_intensities[i] = maximum
        return IPM_intensities

    def get_beam_evt(self, dg="dg1"):
        if self.beamline[dg]['version'] == 3:
            IPM_intensities = self.process_wave8v3(self.beamline[dg]['det_event'])
            Xpos = (IPM_intensities[2] - IPM_intensities[4])/(IPM_intensities[2] + IPM_intensities[4])
            TotInt = np.sum(IPM_intensities)
            return Xpos, TotInt
        else:
            return self.beamline[dg]['det_event'].X_Position(), self.beamline[dg]['det_event'].TotalIntensity()
    
    def get_event_data(self, nevent_start=0, nevent_end=-1):
        self.init_run()
        for nevent,evt in enumerate(self.ds.events()):
            skip_code = self.skip_event(evt,nevent_start,nevent_end)
            if  skip_code < 1:
                self.beamline['event_id'].append(self.num_events['total'])
                self.beamline['time_s'].append(self.get_event_time(evt))
                if 'mirror_pitch' in self.beamline['components']:
                    self.beamline['mirror_pitch']['RBV'].append(self.beamline['mirror_pitch']['det'](evt))
                for dg in ['dg1', 'dg2']:
                    if dg in self.beamline['components']:
                        x_position, total_intensity = self.get_beam_evt(dg)
                        self.beamline[dg]['Xpos'].append(x_position)
                        self.beamline[dg]['TotInt'].append(total_intensity)
            elif skip_code > 1:
                break
        print(self.num_events)

    def plot(self, n_start=30):
        #n_start=30 # DG1 first few events are crap
        time_s = self.beamline['time_s'][n_start:]
        if 'mirror_pitch' in self.beamline['components']:
            pitch = self.beamline['mirror_pitch']['RBV'][n_start:]
        if 'dg1' in self.beamline['components']:
            dg1_int = self.beamline['dg1']['TotInt'][n_start:]
            dg1_xps = self.beamline['dg1']['Xpos'][n_start:]
        if 'dg2' in self.beamline['components']:
            dg2_int = self.beamline['dg2']['TotInt'][n_start:]
            dg2_xps = self.beamline['dg2']['Xpos'][n_start:]
        
        fig = plt.figure(figsize=(8,8), dpi=360, layout="constrained")
        gs = GridSpec(5, 5, figure=fig)

        if 'mirror_pitch' in self.beamline['components']:
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title('time (s)')
            ax1.set_ylabel('pitch')
            ax1.scatter(time_s,pitch)
        
            if 'dg1' in self.beamline['components']:
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.set_ylabel('DG1 Xpos')
                ax2.sharex(ax1)
                ax2.tick_params(labelbottom=False)
                ax2.scatter(time_s,dg1_xps, c=pitch)

                ax4 = fig.add_subplot(gs[3, 0])
                ax4.set_ylabel('DG1 Intensity')
                ax4.sharex(ax1)
                ax4.tick_params(labelbottom=False)
                ax4.scatter(time_s,dg1_int, c=pitch)

                ax6 = fig.add_subplot(gs[0, 1])
                ax6.set_title('DG1 Xpos')
                ax6.tick_params(labelleft=False)
                ax6.scatter(dg1_xps,pitch, c=time_s)

                ax8 = fig.add_subplot(gs[0, 3])
                ax8.set_title('DG1 Intensity')
                ax8.sharey(ax6)
                ax8.tick_params(labelleft=False)
                ax8.scatter(dg1_int,pitch, c=time_s)

                if 'dg2' in self.beamline['components']:
                    ax10 = fig.add_subplot(gs[1:3, 1:3])
                    ax10.set_title('colored by pitch')
                    ax10.set_xlabel('DG1 Xpos')
                    ax10.set_ylabel('DG2 Xpos')
                    ax10.scatter(dg1_xps,dg2_xps,c=pitch)
                    
                    ax11 = fig.add_subplot(gs[1:3, 3:5])
                    ax11.set_title('colored by DG1 Intensity')
                    ax11.set_xlabel('DG1 Xpos')
                    ax11.set_ylabel('DG2 Xpos')
                    ax11.sharey(ax10)
                    ax11.scatter(dg1_xps,dg2_xps,c=dg1_int)
                    
                    ax12 = fig.add_subplot(gs[3:5, 1:3])
                    ax12.set_title('colored by DG2 Intensity')
                    ax12.set_xlabel('DG1 Xpos')
                    ax12.set_ylabel('DG2 Xpos')
                    ax12.sharex(ax10)
                    ax12.scatter(dg1_xps,dg2_xps,c=dg2_int)
                    
                    ax13 = fig.add_subplot(gs[3:5, 3:5])
                    ax13.set_xlabel('DG1 Intensity')
                    ax13.set_ylabel('DG2 Intensity')
                    ax13.scatter(dg1_int,dg2_int, c=time_s)
            
            if 'dg2' in self.beamline['components']:
                ax3 = fig.add_subplot(gs[2, 0])
                ax3.set_ylabel('DG2 Xpos')
                ax3.sharex(ax1)
                ax3.tick_params(labelbottom=False)
                ax3.scatter(time_s,dg2_xps, c=pitch)
        
                ax5 = fig.add_subplot(gs[4, 0])
                ax5.set_ylabel('DG2 Intensity')
                ax5.sharex(ax1)
                ax5.tick_params(labelbottom=False)
                ax5.scatter(time_s,dg2_int, c=pitch)
                
                ax7 = fig.add_subplot(gs[0, 2])
                ax7.set_title('DG2 Xpos')
                ax7.sharey(ax6)
                ax7.tick_params(labelleft=False)
                ax7.scatter(dg2_xps,pitch, c=time_s)

                ax9 = fig.add_subplot(gs[0, 4])
                ax9.set_title('DG2 Intensity')
                ax9.sharey(ax6)
                ax9.tick_params(labelleft=False)
                ax9.scatter(dg2_int,pitch, c=time_s)     
        
        fig.suptitle(f'{self.exp} run {self.run}')
        
        plt.show()
