import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def broken_plot(data, metric, output_path, bottom_max, top_min, top_max, ticks_frequency):
  f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.2})
  sns.barplot(x="V", y=metric, hue="pruning", data=data, ax=ax_top)
  sns.barplot(x="V", y=metric, hue="pruning", data=data, ax=ax_bottom)

  # y-axis
  sns.despine(ax=ax_bottom)
  sns.despine(ax=ax_top, bottom=True)
  ax_bottom.set_ylabel(f"{metric}")
  ax_top.set_ylabel("")
  ax_top.set_ylim(top_min, top_max)   
  ax_bottom.set_ylim(0, bottom_max)
  ax_top.set_yticks(np.arange(top_min, top_max, ticks_frequency))

  # x-axis
  ax_top.get_xaxis().set_visible(False) # Remove x-axis from top graph
  ax_bottom.set_xlabel("Nodes")

  ax_bottom.get_legend().remove()
  ax_top.get_legend().remove()
  # then create a new legend and put it to the side of the figure (also requires trial and error)
  ax_top.legend(loc=(1.025, 0.5), title="Pruning algorithm")

  ax = ax_top
  d = .015  # how big to make the diagonal lines in axes coordinates
  kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
  ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
  ax2 = ax_bottom
  kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
  ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

  # plt.savefig(output_path, bbox_inches='tight')
  return f


def execution_time(input_path, output_path, break_axis=True):
  sns.set_theme(style="ticks")
  data = pd.read_csv(input_path)
  data['time'] = data['time'].apply(lambda x: float(x.split(' ')[0]))
  if break_axis:
    fig = broken_plot(data, 'time', output_path, 300, 500, 1500, 250)
  else:
    sns_plot = sns.barplot(x="V", y="time", hue="pruning", data=data)
    fig = sns_plot.get_figure()

  fig.savefig(output_path, bbox_inches='tight')


def mean_vs_median(input_path, output_paths):
  """
  Compare SHD and SID score between mean and median. Comparison across [0.05, 0.1, 0.15, 0.2] threshold values
  """
  sns.set_theme(style="ticks")
  data = pd.read_csv(input_path)
  data['SHD'] = data['SHD'].apply(lambda x: float(x.split(' ')[0]))
  data['SID'] = data['SID'].apply(lambda x: float(x.split(' ')[0]))

  g = sns.catplot(data=data, x='V', y='SHD', hue='Statistics', col='Threshold', kind='bar')
  g.set(ylim=(0, 300))

  h = sns.catplot(data=data, x='V', y='SID', hue='Statistics', col='Threshold', kind='bar')

  def save_figure(sns_plot, path):
    fig = sns_plot.get_figure()
    fig.save_fig(path)

  save_figure(g, output_paths[0])
  save_figure(h, output_paths[1])


def metric_comparison(input_path, output_path, metric='SHD', break_axis=True):
  """
  Compare SHD/SID metric for Fast, CAM, FastCAM pruning algorithms
  """
  sns.set_theme(style="ticks")
  data = pd.read_csv(input_path)
  data[metric] = data[metric].apply(lambda x: float(x.split(' ')[0]))
  
  if break_axis:
    fig = broken_plot(data, metric, output_path, 300, 500, 5500, 1000)
  else:
    sns_plot = sns.barplot(x="V", y=metric, hue="pruning", data=data)
    fig = sns_plot.get_figure()

  fig.savefig(output_path, bbox_inches='tight')