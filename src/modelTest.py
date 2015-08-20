
import datetime,time
import simplejson as json
import pickle
import numpy as np
from pyspark import SparkConf, SparkContext
import sys,os
import ConfigParser
import logging
from sklearn.metrics import roc_auc_score, roc_curve, auc

from pyspark.sql import *
# conf_path = "/home/hadoop/model.conf"
sc = SparkContext()
conf_path = sys.argv[1]

sqlContext = SQLContext(sc)
time_format = "%Y%m%d_%H-%M-%S"

#========================= 1:set log and get conf =====================
timeStampForRst = time.strftime( time_format, time.localtime())
setlog(timeStampForRst)
logging.info( conf_path )

cf = ConfRead(conf_path)

def getPara(paraName,section="PARA"):
    try:
        return cf.get(section,paraName)
    except Exception, e:
        logging.error("Fail to get para[%s]: %s" % (paraName, e))
        return None
        # sys.exit(1)

code_path = getPara("code_path")
logging.info(code_path)
sc.addPyFile(code_path+'/preprocess/FeatureManager.py')
sc.addPyFile(code_path+'/preprocess/warm_start.py')
sc.addPyFile(code_path +'/preprocess/__init__.py')
sc.addPyFile(code_path+'/optimize/olbfgs.py')
sc.addPyFile(code_path+'/optimize/__init__.py')
sc.addPyFile(code_path+'/trainer.py')
sc.addPyFile(code_path+'/__init__.py')

from FeatureManager import *
from olbfgs import *
from warm_start import set_first_intercept, set_first_intercept_spark

#####################################
data_path = cf.get("PARA","data_path")

max_iter = int(getPara('max_iter'))
min_iter = int(getPara('min_iter'))
grad_norm2_threshold = float(cf.get("PARA","grad_norm2_threshold"))
m_1 = float(cf.get("PARA","m_1"))
c_1 = float(cf.get("PARA","c_1"))
lamb_const_1 = float(cf.get("PARA","lamb_const_1"))
work_path = getPara('work_path')
rst_path = work_path + '/' + timeStampForRst + '/'
os.system('mkdir -p %s' % rst_path )
#=================== 2: create an instance ===============================
feat_path = getPara('feat_path')
feat_file = getPara('feat_file')
t_lines = open(feat_path+'/'+feat_file,'rb').read().splitlines()
single_features = map( lambda x: x.strip(),t_lines[0].split(',') )
logging.info("single_features:%s" % single_features)
#get quad features
quad_arr = t_lines[1:]
quads = map( get_quads, quad_arr )
logging.info("quads: %s" % quads)

hash_feats = HashFeatureManager() \
    .set_k(21) \
    .set_label('target') \
    .set_single_features(single_features) \
    .set_quad_features(quads)

###################################
def get_data(data_path):
    # data_path = getPara('data_new_path')
    parquetFile = sqlContext.load(
            path=data_path,
            source="parquet",
            mergeSchema="false")
    pos_rows = parquetFile.filter(parquetFile.target == 1.0)
    neg_rows = parquetFile.filter(parquetFile.target == 0.0)

    r_subsample = 0.05
    resampled_rows = pos_rows.unionAll(neg_rows.sample(False, r_subsample, 42))

    resampled_json_rows = resampled_rows.toJSON().repartition(sc.defaultParallelism)
    resampled_dict_rows = resampled_json_rows.map(json.loads)

    X = resampled_dict_rows.map(hash_feats.parse_row).cache()
    return X

evaluate_model_path = getPara('evaluate_model_path') # '/mnt1/ctr_model.p.feat6.20150731_20-52-03'
# evaluate_model_path = '/mnt1/new_model'
w_g_l_new = pickle.load(open(evaluate_model_path,'rb'))
w_new = w_g_l_new[0]

test_data = getPara('test_data') #'s3n://kiip-reports/joined_rms_2_5/2015/07/01/'
X_test = get_data( test_data )

verify_pred = X_test.map(
    lambda x: (x[0], lr_predict(w_new, x[1]))).collect() 

roc = roc_auc_score(
    [x[0] for x in verify_pred],
    [x[1] for x in verify_pred])
print roc


