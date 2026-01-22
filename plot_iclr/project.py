import os, sys
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.font_manager
# matplotlib.font_manager._rebuild()

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

results_iclr_path = os.path.join(current_path, 'results', 'GeoDiff')
if not os.path.exists(results_iclr_path):
    os.makedirs(results_iclr_path, exist_ok=True)
