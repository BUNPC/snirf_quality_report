import mne
import mne_nirs
import os
from pathlib import Path
import pandas as pd
import pysnirf2

from mne.io import read_raw_nirx, read_raw_snirf
from mne_nirs.io import write_raw_snirf
from numpy.testing import assert_allclose

import numpy as np
from itertools import compress
import matplotlib.pyplot as plt

from mne.preprocessing.nirs import optical_density
from mne_nirs.preprocessing import peak_power, scalp_coupling_index_windowed
from mne_nirs.visualisation import plot_timechannel_quality_metric

class snirf_quality_report():

	def plot_raw(self, raw, report):
	#     logger.debug("    Creating raw plot")
	    fig1 = raw.plot(n_channels=len(raw.ch_names),
	                    duration=raw.times[-1],
	                    show_scrollbars=False, clipping=None)

	    msg = "Plot of the raw signal"
	    report.add_figure(fig=fig1, caption=msg, title="Raw Waveform")
	    plt.close()

	    return raw, report

	def summarise_triggers(self, raw, report):
	#     logger.debug("    Creating trigger summary")

	    events, event_dict = mne.events_from_annotations(raw, verbose=False)
	    print('before fig')
	    fig2 = mne.viz.plot_events(events, event_id=event_dict,sfreq=raw.info['sfreq'])
	    print('after fig')
	    report.add_figure(fig=fig2, title="Triggers")
	    plt.close()

	    return raw, report

	def summarise_odpsd(self, raw, report):
	#     logger.debug("    Creating PSD plot")

	    fig, ax = plt.subplots(ncols=2, figsize=(15, 8))

	    raw.plot_psd(ax=ax[0])
	    raw.plot_psd(ax=ax[1], average=True)
	    ax[1].set_title("Average +- std")

	    msg = "PSD of the optical density signal."
	    report.add_figure(fig=fig, title="OD PSD", caption=msg)
	    plt.close()
	    return raw, report


	def summarise_sci_window(self, raw, report, threshold=0.8):
	#     logger.debug("    Creating windowed SCI summary")

	    if raw.info['lowpass'] < 1.5:
	        print("SCI calculation being run over limited frequency range, check if valid.")
	        h_trans_bandwidth = 0.2
	        h_freq = 1.5 - raw.info['lowpass'] - 0.05
	    else:
	        h_freq = 1.5
	        h_trans_bandwidth = 0.3


	    _, scores, times = scalp_coupling_index_windowed(raw, time_window=60, h_freq=h_freq, h_trans_bandwidth=h_trans_bandwidth)
	    fig = plot_timechannel_quality_metric(raw, scores, times,
	                                          threshold=threshold,
	                                          title="Scalp Coupling Index "
	                                          "Quality Evaluation")
	    msg = "Windowed SCI."
	    report.add_figure(fig=fig, title="SCI Windowed", caption=msg)

	    return raw, report

	def summarise_pp(self, raw, report, threshold=0.8):
	#     logger.debug("    Creating peak power summary")

	    if raw.info['lowpass'] < 1.5:
	        print("PP calculation being run over limited frequency range, check if valid.")
	        h_trans_bandwidth = 0.2
	        h_freq = 1.5 - raw.info['lowpass'] - 0.05
	    else:
	        h_freq = 1.5
	        h_trans_bandwidth = 0.3

	    _, scores, times = peak_power(raw, time_window=10, h_freq=h_freq, h_trans_bandwidth=h_trans_bandwidth)
	    fig = plot_timechannel_quality_metric(raw, scores, times,
	                                          threshold=threshold,
	                                          title="Peak Power "
	                                          "Quality Evaluation")
	    msg = "Windowed Peak Power."
	    report.add_figure(fig=fig, title="Peak Power", caption=msg)

	    return raw, report

	def summarise_sci(self, raw, report, threshold=0.8):
	#     logger.debug("    Creating SCI summary")

	    if raw.info['lowpass'] < 1.5:
	        print("SCI calculation being run over limited frequency range, check if valid.")
	        h_trans_bandwidth = 0.2
	        h_freq = 1.5 - raw.info['lowpass'] - 0.05
	    else:
	        h_freq = 1.5
	        h_trans_bandwidth = 0.3

	    sci = mne.preprocessing.nirs.scalp_coupling_index(raw, h_freq=h_freq, h_trans_bandwidth=h_trans_bandwidth)
	    raw.info['bads'] = list(compress(raw.ch_names, sci < threshold))

	    fig, ax = plt.subplots()
	    ax.hist(sci)
	    ax.set(xlabel='Scalp Coupling Index', ylabel='Count', xlim=[0, 1])
	    ax.axvline(linewidth=4, color='r', x=threshold)

	    msg = f"Scalp coupling index with threshold at {threshold}." \
	          f"Results in bad channels {raw.info['bads']}"
	    report.add_figure(fig=fig, caption=msg, title="Scalp Coupling Index")

	    return raw, report, sci

	def summarise_montage(self, raw, report):
	#     logger.debug("    Creating montage summary")
	    fig3 = raw.plot_sensors()
	    msg = f"Montage of sensors." \
	          f"Bad channels are marked in red: {raw.info['bads']}"
	    report.add_figure(fig=fig3, title="Montage", caption=msg)

	    return raw, report

	def is_valid_bids_path_for_snirf_file(self, snirf_path):
		norm_path = os.path.normpath(snirf_path)
		all_dirs = norm_path.split(os.sep)
		return_idx = 0

		if len(all_dirs) >= 5 and all_dirs[-3].startswith('ses-') and all_dirs[-4].startswith('sub-'):
		    return_idx = -4
		elif len(all_dirs) >= 4 and all_dirs[-3].startswith('sub-'):
		    return_idx = -3

		return return_idx

	def run_report(self, snirf_path = None, path_to_save_report = None, filename_to_save = None):

		if snirf_path is None:
			return 'Please pass path to the snirf file'

		report = mne.Report(verbose=True, raw_psd=True)
		snirf_intensity = read_raw_snirf(snirf_path)
		raw, report = self.plot_raw(snirf_intensity, report)
		# raw, report = self.summarise_triggers(snirf_intensity, report)
		raw = optical_density(snirf_intensity)
		# raw, report = self.summarise_odpsd(raw, report)
		raw, report = self.summarise_sci_window(raw, report, threshold=0.6)
		raw, report = self.summarise_pp(raw, report, threshold=0.1)
		raw, report, sci = self.summarise_sci(raw, report, threshold=0.6)
		raw, report = self.summarise_montage(raw, report)
		if path_to_save_report:
			if os.path.isdir(path_to_save_report):
				if filename_to_save:
					filename_and_path = os.path.join(path_to_save_report, filename_to_save)
				else:
					filename_and_path = os.path.join(path_to_save_report, 'quality_report.html')
				report.save(filename_and_path, overwrite=True, open_browser=False)
		else:
			return_idx = self.is_valid_bids_path_for_snirf_file(snirf_path)
			if return_idx!= 0:
				norm_path = os.path.normpath(snirf_path)
				all_dirs = norm_path.split(os.sep)
				if all_dirs[return_idx+1].startswith('ses-'):
					report_dir = os.path.join(*all_dirs[:return_idx],'derivatives',all_dirs[return_idx],all_dirs[return_idx+1],'report')
				else:
					report_dir = os.path.join(*all_dirs[:return_idx],'derivatives',all_dirs[return_idx],'report')
					
				report_dir = Path('/'+report_dir)

				# print(report_dir)

				if not os.path.isdir(report_dir):
					os.makedirs(report_dir)
			
				report_path = os.path.join(report_dir,'quality_report.html')
				report.save(report_path, overwrite=True, open_browser=False)
				
				channels_path = snirf_path.replace('_nirs.snirf', '_channels.tsv')
				channels_path_ext = os.path.splitext(channels_path)[-1].lower()
				if os.path.isfile(channels_path) and channels_path_ext=='.tsv':
				    channels_df = pd.read_csv(channels_path, sep='\t')
				    if 'name' in channels_df.keys():
				        channel_names = channels_df['name']
				        sci_tsv = []
				        pp_tsv = []
				        
				        for channel_name in channel_names:
				            if channel_name in raw.ch_names:
				                idx = raw.ch_names.index(channel_name)
				                sci_tsv.append(sci[idx])
				                pp_tsv.append('')
				
				            else:
				                sci_tsv.append('')
				                pp_tsv.append('')
				
				        channels_df['scalp_coupling_index'] = sci_tsv
				        channels_df['peak_power'] = pp_tsv
				
				        channel_filename = os.path.split(channels_path)[1]
				        channel_tsv_path_to_save = os.path.join(report_dir,channel_filename)
				        print(channel_filename)
				        print(channel_tsv_path_to_save)
				        channels_df.to_csv(channel_tsv_path_to_save, sep="\t")
					
		# plt.close()
