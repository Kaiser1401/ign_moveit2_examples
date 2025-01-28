import dill
import numpy as np
from copy import deepcopy

from pose_sample import add_noise_to_pose_msg
from geometry_msgs.msg import PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped, Pose
from math3d import Transform, Orientation, PositionVector, Versor, Vector
from math3d.visualization import TransformVisualizer

import numpy.typing as npt
from pathlib import Path

import json
from scipy.spatial.transform import Rotation as SpR
from scipy.stats import circmean, circstd
from dataclasses import dataclass
from numpy.typing import ArrayLike


def pose_distance(p1:Pose|Transform,p2:Pose|Transform,m_per_rad=0):
    if isinstance(p1,Pose):
        t1 = p2t(p1)
    else:
        t1 = p1
    if isinstance(p2,Pose):
        t2 = p2t(p2)
    else:
        t2 = p2

    t_diff = t1.inverse * t2
    #assert isinstance(t_diff, Transform)
    dist_rotation = t_diff.orientation.ang_norm * m_per_rad
    dist_linear = t_diff.pos.length
    return dist_rotation + dist_linear

def joint_dist(j1:list|npt.ArrayLike,j2:list|npt.ArrayLike):
    if isinstance(j1,list):
        a1 = np.array(j1)
    else:
        a1 = j1

    if isinstance(j2,list):
        a2 = np.array(j2)
    else:
        a2 = j2

    return np.linalg.norm(a1-a2)

class DataEntry_v1(object):
    def __init__(self):
        self.start_offset_is = Transform()
        self.goal_offset_is = Transform()
        self.start_common = Transform()
        self.sampled_offset = Transform()
        self.sampled_variance = 6*[0]
        self.b_simulated = False
        self.b_prediction = None
        self.b_outcome = None

    def get_pose_is(self):
        return self.start_common * self.start_offset_is

    def get_goal_is(self):
        return self.get_pose_is() * self.goal_offset_is

    def get_pose_hat(self):
        return self.start_common * self.start_offset_is * self.sampled_offset

    def get_goal_hat(self):
        return self.get_pose_hat() * self.goal_offset_is



class DataEntry(DataEntry_v1):
    def __init__(self):
        super().__init__()
        # remember the final pose, makes reevaluating on different success criteria possible
        self.final_pose_from_goal = Transform()
        self.b_handling_error_likely = False # have some heuristic that marks failures due to planning / robot / simulation error

    def set_final_is(self, pose_final):
        pose_final_shall = self.get_goal_is()
        self.final_pose_from_goal = pose_final_shall.inverse * pose_final



def p2t(p:Pose)-> Transform:
    q=Versor(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z) # w,x,y,v
    t=Transform(q.orientation,PositionVector(p.position.x, p.position.y, p.position.z))
    return t

def t2p(t:Transform)->Pose:
    p = Pose()
    p.orientation.w = t.orientation.versor.scalar_part
    p.orientation.x = t.orientation.versor.vector_part[0]
    p.orientation.y = t.orientation.versor.vector_part[1]
    p.orientation.z = t.orientation.versor.vector_part[2]
    p.position.x = t.pos[0]
    p.position.y = t.pos[1]
    p.position.z = t.pos[2]
    return p



def sample_covar_diag(variance_for_covar_diag=6*[0]):

    sigma = np.sqrt(variance_for_covar_diag)
    new_sigma = np.random.normal(np.zeros(len(sigma)), np.array(sigma))
    new_cov = np.power(new_sigma, 2)
    return new_cov


def sample_xyz_rpy_zero_mean(covar_diag = 6*[0], covarMtx=None):
    if covarMtx is None:
        covarMtx = np.diag(covar_diag)

    noise = np.random.multivariate_normal(mean=np.zeros(6), cov=covarMtx)

    # covariance rotations in ROS seem to be Fixed-Frame (extrinsic) xyz
    rot = Orientation.new_from_euler(noise[3:6],encoding='xyz')
    pos = PositionVector(noise[0:3])
    t = Transform(rot, pos)

    pose_sampled = PoseWithCovariance()
    pose_sampled.pose = t2p(t)
    pose_sampled.covariance = covarMtx.flatten().tolist()

    return pose_sampled, t


def create_samples_zero_mean(sample_variance=None, count=10, append_to:list = None, sample_variance_each_time=False) -> list[DataEntry]:

    if append_to is None:
        append_to = []

    if sample_variance is None:
        sample_variance = 6 * [0]

    original_variance = deepcopy(sample_variance)
    data = append_to


    for i in range(count):
        if sample_variance_each_time:
            sample_variance = sample_covar_diag(original_variance)
        _, t_sampled = sample_xyz_rpy_zero_mean(sample_variance)

        entry = DataEntry()
        entry.sampled_offset = t_sampled
        entry.sampled_variance = sample_variance

        data.append(entry)

    return data


def write_data(data:list, fn, backup=False):
    p = Path(fn)
    ptmp = p.with_suffix('.tmp')
    with open(str(ptmp), 'wb') as file:
        dill.dump(data, file)
    if p.exists():
        if backup:
            i = 0
            pb = p.with_suffix(f'.backup_{i}')
            while pb.exists():
                i+=1
                pb = p.with_suffix(f'.backup_{i}')
            p.rename(pb)
        else:
            p.unlink()
    ptmp.rename(p)


def load_data(fn)->list:
    with open(fn, 'rb') as file:
        data = dill.load(file)
    return data



#--------- Ahmeds data

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data['rotation_matrix']), np.array(data['translation_matrix'])


def rot_trans_to_T(rot,trans):
    T = Transform()
    T.set_pos(Vector(trans))
    T.set_orient(Orientation(rot))
    return T

@dataclass
class TransformStats:
    poseCount: int = 0
    TGroundTruth: Transform = None
    TAvg: Transform = None
    stdDev_euler = ArrayLike
    stdDev_position = ArrayLike




def create_entry_from_realworld_json(fn_json):
    entry = DataEntry()

    rots, trans = read_json_file(fn_json)
    trans /= 1000  # mm to m
    Ts = []
    Tgt = None
    for i in range(len(rots)):
        T = rot_trans_to_T(rots[i], trans[i])
        if i == len(rots) - 1:
            Tgt = T
        else:
            Ts.append(T)

    tStats = transformation_averaging(Ts)
    tStats.TGroundTruth = Tgt

    Tdiff = Tgt.inverse * tStats.TAvg

    entry.sampled_offset = Tdiff
    entry.sampled_variance = np.power(np.concatenate((tStats.stdDev_position,tStats.stdDev_euler)),2)

    return entry

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.mean.html
# https://users.cecs.anu.edu.au/~hartley/Papers/PDF/Hartley-Trumpf:Rotation-averaging:IJCV.pdf
def transformation_averaging(listOfT):

    ddof = 1 # -> sample standard deviation (../(N-1))-> https://numpy.org/doc/2.1/reference/generated/numpy.std.html
    # treat pos and orientation separate
    c = len(listOfT)
    if c == 1:
        return listOfT[0]

    pos = []
    eulerDiff = []
    rStacked = np.zeros((c+1,3,3))
    for i in range(c):
        t = listOfT[i]
        assert isinstance(t, Transform)
        pos.append(t.get_pos().array)
        rStacked[i]=t.get_orient().matrix


    spRs = SpR.from_matrix(rStacked)
    oAvg = Orientation(spRs.mean().as_matrix())


    pAvg = np.array(pos).mean(0)
    pStd = np.array(pos).std(0,ddof=ddof)

    #get orientation_differences from average in euler


    for i in range(c):
        t = listOfT[i]
        oDiff = oAvg.get_inverse() * t.orient
        eulerDiff.append(oDiff.to_euler('xyz'))

    #eAvgC = circmean(np.array(eulerDiff), axis=0)
    #eStdC = circstd(np.array(eulerDiff), axis=0)
    eAvg = np.array(eulerDiff).mean(0)
    eStd = np.array(eulerDiff).std(0,ddof=ddof)

    eMaxAbs = np.max(np.abs(np.array(eulerDiff)),axis=0)

    print(f'pMean: {pAvg}')
    print(f'pStdD: {pStd}')

    print(f'eMax: {eMaxAbs}')
    print(f'eMean: {eAvg}')
    print(f'eStdD: {eStd}')
    #print(f'eMeanC: {eAvgC}')
    #print(f'eStdDC: {eStdC}')

    tstat = TransformStats()
    tstat.TAvg = Transform(oAvg,Vector(pAvg))
    tstat.poseCount = c
    tstat.stdDev_euler = eStd
    tstat.stdDev_position = pStd

    return tstat


def vizT(T,tv=None,veclen=1.0):
    if tv is None:
        tv = TransformVisualizer(plot_identity=True, identity_label='O',uvec_length=veclen)
    # tv.plot(m3d.Transform(), label='Base')
    if (type(T) is list):
        c=0
        for e in T:
            tv.plot(e, uvec_length=veclen, label=f'T_{c}')
            c+=1
    elif (type(T) is dict):
        for k in T.keys():
            tv.plot(T[k], uvec_length=veclen, label=str(k))
    else:
        tv.plot(T,uvec_length=veclen,label='Transformed')
    return tv

def get_realworld_entries_from_folder(folderpath):
    folder = Path(folderpath)
    data = []
    i = 0
    for jf in folder.rglob('*.json'):
        print('-------')
        print(f'{i}: {jf.name}')
        entry = create_entry_from_realworld_json(jf)
        print(entry.sampled_offset)
        print(entry.sampled_variance)
        data.append(entry)
        i+=1

    return data



