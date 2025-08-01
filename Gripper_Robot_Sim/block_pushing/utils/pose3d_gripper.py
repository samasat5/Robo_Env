
"""A simple 6DOF pose container.
"""

import dataclasses
import numpy as np
from scipy.spatial import transform


class NoCopyAsDict(object):
    """Base class for dataclasses. Avoids a copy in the asdict() call."""

    def asdict(self):
        """Replacement for dataclasses.asdict.

        TF Dataset does not handle dataclasses.asdict, which uses copy.deepcopy when
        setting values in the output dict. This causes issues with tf.Dataset.
        Instead, shallow copy contents.

        Returns:
          dict containing contents of dataclass.
        """
        return {k.name: getattr(self, k.name) for k in dataclasses.fields(self)}


@dataclasses.dataclass
class Pose3d_gripper(NoCopyAsDict):
    """Container for left and right finger 6DoF poses."""

    orientation_left: transform.Rotation
    orientation_right: transform.Rotation
    translation_left: np.ndarray
    translation_right: np.ndarray

    @property
    def vec7(self):
        """Return 14D vector: [tL, tR, qL, qR]"""
        return np.concatenate([
        self.translation_left,
        self.translation_right,
        self.orientation_left.as_quat(),
        self.orientation_right.as_quat()
    ])  # this gives [xL, yL, zL, xR, yR, zR, qxL, qyL, qzL, qwL, qxR, qyR, qzR, qwR]


    def serialize(self):
        return {
            "orientation_left": self.orientation_left.as_quat().tolist(),
            "orientation_right": self.orientation_right.as_quat().tolist(),
            "translation_left": self.translation_left.tolist(),
            "translation_right": self.translation_right.tolist(),
        }

    @staticmethod
    def deserialize(data):
        return Pose3d_gripper(
            orientation_left=transform.Rotation.from_quat(data["orientation_left"]),
            orientation_right=transform.Rotation.from_quat(data["orientation_right"]),
            translation_left=np.array(data["translation_left"]),
            translation_right=np.array(data["translation_right"]),
        )

    def __eq__(self, other):
        return np.array_equal(self.orientation_left.as_quat(), other.orientation_left.as_quat()
        ) and np.array_equal(self.orientation_right.as_quat(), other.orientation_right.as_quat()
        ) and np.array_equal(self.translation_left, other.translation_left
        ) and np.array_equal(self.translation_right, other.translation_right)


    def __ne__(self, other):
        return not self.__eq__(other)
