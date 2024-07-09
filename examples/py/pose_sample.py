import numpy as np
from geometry_msgs.msg import PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped
from scipy.spatial.transform import Rotation

def so3_hat(theta_vec):
    """Note: adapted from pylie."""
    return np.array([[0, -theta_vec[2], theta_vec[1]],
                     [theta_vec[2], 0, -theta_vec[0]],
                     [-theta_vec[1], theta_vec[0], 0]])


def se3_hat(xi_vec):
    """Note: adapted from pylie."""
    return np.block([[so3_hat(xi_vec[3:]), xi_vec[:3, None]], [np.zeros((1, 4))]])


def se3_exp(xi_vec):
    """Note: adapted from pylie. To avoid this code, could consider using pylie or manif Python packages to do this for us."""
    xi_hat = se3_hat(xi_vec)
    theta = np.linalg.norm(xi_vec[3:])

    tmp = np.eye(4) + xi_hat

    if theta < 1e-10:
        return tmp
    else:
        tmp2 = ((1 - np.cos(theta)) / (theta ** 2)) * np.linalg.matrix_power(xi_hat, 2) + \
            ((theta - np.sin(theta)) / (theta ** 3)) * np.linalg.matrix_power(xi_hat, 3)
        return tmp + tmp2


def sample_gaussian_se3(mean, covar):
    """Samples a pose from a Gaussian distribution on SE(3) as defined in "A micro Lie theory for state estimation in robotics".
    For the variance on the rotation elements this can be interpreted as sampling a rotation vector, meaning the units are in radians.
    """
    noise = np.random.multivariate_normal(mean=np.zeros(6), cov=covar)  # sample zero-mean Gaussian on SE3 tangent
    noisy_pose = mean @ se3_exp(noise)  # map onto SE3 and compose with mean
    return noisy_pose


def add_noise_to_pose_msg(pose, variance= 6 *[0]):
    covar = np.diag(variance)
    rot_mat = Rotation.from_quat([pose.orientation.x, pose.orientation.y,
                                  pose.orientation.z, pose.orientation.w]).as_matrix()
    pos = np.array([pose.position.x, pose.position.y, pose.position.z])
    pose_mat = np.block([[rot_mat, pos.reshape(3, 1)], [0, 0, 0, 1]])

    noisy_pose_mat = sample_gaussian_se3(pose_mat, covar)
    noisy_quat = Rotation.from_matrix(noisy_pose_mat[:3, :3]).as_quat()

    pose_dist = PoseWithCovariance()
    pose_dist.pose.position.x = noisy_pose_mat[0, 3]
    pose_dist.pose.position.y = noisy_pose_mat[1, 3]
    pose_dist.pose.position.z = noisy_pose_mat[2, 3]
    pose_dist.pose.orientation.x = noisy_quat[0]
    pose_dist.pose.orientation.y = noisy_quat[1]
    pose_dist.pose.orientation.z = noisy_quat[2]
    pose_dist.pose.orientation.w = noisy_quat[3]
    pose_dist.covariance = covar.flatten().tolist()

    return pose_dist
