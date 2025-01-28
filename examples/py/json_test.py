import data_utils
import classify

fndata = '/home/klaus/code/pymi2_ws/sim_data/out/jsondata_nt44_v2_6.dill'
fn_classifier=  '/home/klaus/code/pymi2_ws/sim_data/out/jsondata_nt44_v2_6.classifier'
import analyse_data as ad

clf = data_utils.load_data(fn_classifier)

assert isinstance(clf, classify.Classifyer)
ad.confusion_data_binary(clf.confusion,True)

data = data_utils.load_data(fndata)

for e in data:
    assert isinstance(e, data_utils.DataEntry)
    print(e.b_simulated, e.b_prediction,e.b_outcome, e.sampled_variance)