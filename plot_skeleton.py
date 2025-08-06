import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

keypoints = [
    "Hips", "RHip", "RKnee", "RAnkle", "RFoot", "LHip", "LKnee", "LAnkle", "LFoot",
    "Spine", "Neck", "Head", "RShoulder", "RElbow", "RHand", "LShoulder", "LElbow", "LHand"
]

skeleton = [
    [1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [6, 7], [7, 8], [8, 9],
    [1, 10], [10, 11], [11, 12], [11, 13], [13, 14], [14, 15], [11, 16], [16, 17], [17, 18]
]

points = [
      [
        -7828.30861339466,
        -156.82589131472963,
        965.4331178900977
      ],
      [
        -7903.648186381374,
        -52.7338815404916,
        913.6099667174071
      ],
      [
        -7959.701568876527,
        -60.309925066010514,
        514.0479239363847
      ],
      [
        -7891.616449398725,
        8.829056126459824,
        99.29168034104171
      ],
      [
        -8043.711720743789,
        -16.193242494592713,
        38.55909759213956
      ],
      [
        -7800.448753805973,
        -306.8370527360755,
        921.0205525248447
      ],
      [
        -7849.893127493523,
        -314.51718884067054,
        498.43691996723715
      ],
      [
        -7788.381168137886,
        -276.9892291006988,
        96.2487838803884
      ],
      [
        -7939.442940443511,
        -389.94744250852864,
        33.75697202684038
      ],
      [
        -7881.947956021886,
        -164.86192423364844,
        1171.357876174908
      ],
      [
        -8005.147720316092,
        -196.01878837606387,
        1429.091844699677
      ],
      [
        -8073.085654325535,
        -184.64770564668015,
        1572.874086786259
      ],
      [
        -7991.008868057607,
        -26.20232712716356,
        1358.865055008149
      ],
      [
        -8000.236179591059,
        64.05561633513123,
        1074.482113209617
      ],
      [
        -8282.200780330371,
        -65.12289953950247,
        897.4610207428403
      ],
      [
        -7987.197792415478,
        -349.59706514308255,
        1403.2462567870557
      ],
      [
        -7983.821247301892,
        -387.72231481050903,
        1135.4513719658362
      ],
      [
        -8250.410682932275,
        -259.43456536477134,
        887.6297531616578
      ]
    ]

points = list(zip(*points))  # Transpose to get x, y, z lists

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter all points in blue except Head (index 11)
ax.scatter(points[0][:11] + points[0][12:], points[1][:11] + points[1][12:], points[2][:11] + points[2][12:], c='b', marker='o')
# Scatter Head point in red
ax.scatter(points[0][11], points[1][11], points[2][11], c='r', marker='o', s=60, label='Head')

# Draw skeleton connections
for joint in skeleton:
    i, j = joint[0] - 1, joint[1] - 1  # Convert to 0-based index
    ax.plot([points[0][i], points[0][j]],
            [points[1][i], points[1][j]],
            [points[2][i], points[2][j]], c='k')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

z_min, z_max = min(points[2]), max(points[2])
center = (z_min + z_max) / 2
scale = (z_max - z_min) * 0.5  # Stretch factor (1.5x)
ax.set_zlim(center - scale/2, center + scale/2)

plt.title('3D Skeleton Plot')
plt.legend()
plt.show()