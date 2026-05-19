from bsor.Bsor import make_bsor
import quaternion
import numpy as np
from util.dtype import BeatSketchBlock, BeatSketchTrackingData
from util import get_bpm_for_song

# TODO: Figure out what the base vector is (unit vector in which direction?)
# This is almost certainly correct
base_vec = np.array([1, 0, 0])
angle_comp_vec = base_vec
# TODO: Verify I got this right
translation = [0, 4, 2, 6, 1, 7, 3, 5]


# https://github.com/BeatLeader/BS-Open-Replay
# This is the replay format used
def load_replay_data(file: str):
    """Generate training data from the specified replay file
       The bpm for the song is fetched automatically from the BeatSaver API

    Args:
        file: Path to the BSOR file to process
    """
    with open(file, "rb") as f:
        # TODO: Check if rotation is correct (or if controller offsets need to be computed)
        # bsor.controller_offsets.left - This is how to get the offsets if needed
        # I am almost certain this is correct
        bsor = make_bsor(f)

        bpm = get_bpm_for_song(bsor.info.songHash)

        # Discard map criteria
        if bsor.info.speed != 0:
            print("This map was played in practice mode -> Not suitable")
            exit(0)
        tracking_data: list[BeatSketchTrackingData] = []

        for frame in bsor.frames:
            hand_l = frame.left_hand
            hand_r = frame.left_hand
            quat_l = quaternion.quaternion(
                hand_l.w_rot, hand_l.x_rot, hand_l.y_rot, hand_l.z_rot
            )
            quat_r = quaternion.quaternion(
                hand_r.w_rot, hand_r.x_rot, hand_r.y_rot, hand_r.z_rot
            )
            dir_l = quaternion.rotate_vectors(quat_l, base_vec)
            dir_r = quaternion.rotate_vectors(quat_r, base_vec)
            tip_l = dir_l + np.array(hand_l.position)
            tip_r = dir_r + np.array(hand_l.position)
            tracking_data.append(
                {
                    "left": np.concatenate((tip_l, dir_l)),
                    "right": np.concatenate((tip_r, dir_r)),
                    "time": frame.time,
                }
            )

        block_data: list[BeatSketchBlock] = []
        # bsor.notes.reverse()
        for block in bsor.notes:
            if block.cut:
                # Do cheapo projection (just setting the z axis to 0) to compute the cut angle using a [0, 1, 0] vector
                adjusted_angle = compute_angle(np.array(block.cut.cutNormal))
                orientation = orientation_from_angle(adjusted_angle)
                block_data.append(
                    {
                        "is_right_hand": block.colorType == 1,
                        "orientation": orientation,
                        "time": block.event_time - block.cut.timeDeviation,
                        "exact_time": block.event_time,
                        "x": block.lineIndex,
                        "y": block.noteLineLayer,
                        "good_cut": True,
                    }
                )
            else:
                block_data.append(
                    {
                        "good_cut": False,
                        "orientation": 0,
                        "time": block.event_time,
                        "exact_time": block.event_time,
                        "is_right_hand": False,
                        "x": 0,
                        "y": 0,
                    }
                )
    return tracking_data, block_data, bpm, bsor.info.jumpDistance


def orientation_from_angle(angle: float) -> int:
    loc = int(((angle + 22.5) % 360) // 45)

    return translation[loc]


def compute_angle(vec: np.ndarray):
    # Do cheapo projection (just setting the z axis to 0) to compute the cut angle using a [0, 1, 0] vector
    vec[2] = 0
    angle = np.arccos(vec.dot(angle_comp_vec) / (np.linalg.norm(vec))) / np.pi * 180
    cross = np.linalg.cross(angle_comp_vec, vec)
    left_side = cross[2] < 0
    return 360 - angle if left_side else angle
