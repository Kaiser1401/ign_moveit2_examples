from fontTools.ttLib.tables.grUtils import entries
import random
import data_utils

target_from_start_offset = data_utils.Transform()
#cube:
#target_from_start_offset.pos[1] =-0.15
#nt39, nt44:
target_from_start_offset.pos[1] =0.20 # move 20cm along y(body)
target_from_start_offset.orientation=data_utils.Orientation.new_from_euler((-1.57,0,0),encoding='xyz') # turn -90deg around x(body)

json_folder = '/home/klaus/code/pymi2_ws/sim_data/json'
entries_out_file = 'jsondata_nt44_v2_6_tmp.dill'

data = data_utils.get_realworld_entries_from_folder(json_folder)

print('------------------')

for e in data:
    e.goal_offset_is = target_from_start_offset


for e in data:
    print(e.sampled_offset, e.sampled_variance)

data_utils.write_data(data,entries_out_file)
