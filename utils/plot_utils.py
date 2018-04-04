import matplotlib
import matplotlib.pyplot as plt

def get_colors():
  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color']
  colors_dict = {}
  colors_dict["exact"] = colors[0]
  colors_dict["fp"] = colors[1]
  for idx, nbit in enumerate([1,2,4,8,16,32] ):
    colors_dict[str(nbit)] = colors[idx + 2]
  colors_dict["pca"] = colors[len(colors_dict.keys() ) ]
  #print colors_dict
  return colors_dict
