
# install package scipy 1.4.1,
# install package numpy-quaternion https://github.com/moble/quaternion
# Author: Mingchuan ZHOU
# Contact: mingchuan.zhou@in.tum.de
import sys
# import open3d as o3d
from python import vrep
import array
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import open3d as o3d
import numpy.polynomial.polynomial as poly
import quaternion
import time

print('Program started')
vrep.simxFinish(-1)  # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
if clientID != -1:
    print('Connected to remote API server')
    res, v0 = vrep.simxGetObjectHandle(clientID, 'Vision_global_rgb', vrep.simx_opmode_oneshot_wait)
    res, v1 = vrep.simxGetObjectHandle(clientID, 'Vision_global_depth', vrep.simx_opmode_oneshot_wait)
    res, v2 = vrep.simxGetObjectHandle(clientID, 'Vision_global_rgb_real', vrep.simx_opmode_oneshot_wait)
    res, trajectory_sphere_ID = vrep.simxGetObjectHandle(clientID, 'trajectory_sphere', vrep.simx_opmode_oneshot_wait)
    res, robot_base_ID = vrep.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_link1', vrep.simx_opmode_oneshot_wait)
    res, Robot_target_ID = vrep.simxGetObjectHandle(clientID, 'Robot_target', vrep.simx_opmode_oneshot_wait)


    res, pos_v0 = vrep.simxGetObjectPosition(clientID, v0, robot_base_ID, vrep.simx_opmode_oneshot_wait)
    res, ori_v0 = vrep.simxGetObjectOrientation(clientID, robot_base_ID, v0, vrep.simx_opmode_oneshot_wait)
    res, qua_v0 = vrep.simxGetObjectQuaternion(clientID,  v0, robot_base_ID,  vrep.simx_opmode_oneshot_wait)
    mat_v0 = quaternion.as_rotation_matrix(np.quaternion(qua_v0[0], qua_v0[1], qua_v0[2], qua_v0[3]))

    res, resolution, image = vrep.simxGetVisionSensorImage(clientID, v0, 0, vrep.simx_opmode_streaming)
    res_depth, resolution_depth, image_depth = vrep.simxGetVisionSensorDepthBuffer(clientID, v1,
                                                                                   vrep.simx_opmode_streaming)
    res, resolution_rgb, image_rgb = vrep.simxGetVisionSensorImage(clientID, v2, 0, vrep.simx_opmode_streaming)
    mask = np.zeros((256, 256), 'float')
    imcount = 0
    while (vrep.simxGetConnectionId(clientID) != -1):
        res, resolution, image = vrep.simxGetVisionSensorImage(clientID, v0, 0, vrep.simx_opmode_buffer)
        res, resolution_rgb, image_rgb = vrep.simxGetVisionSensorImage(clientID, v2, 0, vrep.simx_opmode_buffer)
        if res == vrep.simx_return_ok:
            # res = vrep.simxSetVisionSensorImage(clientID, v1, image, 0, vrep.simx_opmode_oneshot)
            imcount = imcount + 1
            res, resolution, image = vrep.simxGetVisionSensorImage(clientID, v0, 0, vrep.simx_opmode_buffer)
            img = np.array(image, dtype=np.uint8)
            img.resize([resolution[1], resolution[0], 3])
            img = cv2.flip(img, 0)
            res, resolution_rgb, image_rgb = vrep.simxGetVisionSensorImage(clientID, v2, 0, vrep.simx_opmode_buffer)
            img_rgb = np.array(image_rgb, dtype=np.uint8)
            img_rgb.resize([resolution_rgb[1], resolution_rgb[0], 3])
            img_rgb = cv2.flip(img_rgb, 0)
            # img[np.where((img != [16, 0, 0]).all(axis=2))] = [0, 0, 0]
            index = np.where((img == [16, 0, 0]).all(axis=2))
            img[np.where((img == [16, 0, 0]).all(axis=2))] = [255, 255, 255]


            f1 = poly.polyfit(index[0], index[1], 3)
            ffitRound = list(range(round(min(index[0])), round(max(index[0]))))
            # ffit = poly.Polynomial(f1)  # instead of np.poly1d
            ffit = poly.polyval(ffitRound, f1)
            res_depth, resolution_depth, image_depth = vrep.simxGetVisionSensorDepthBuffer(clientID, v1,
                                                                                           vrep.simx_opmode_buffer)
            img_depth = np.array(image_depth, dtype=np.float)
            img_depth.resize([resolution_depth[1], resolution_depth[0], 1])
            img_depth = cv2.flip(img_depth, 0)
            img_depth_0_1 = img_depth.copy()
            img_depth = img_depth*(2-0.01)+0.01

            d_points = []
            point = []
            # point = np.zeros(shape=(1, 3))
            for y in range(0, resolution[1]):
                for x in range(0, resolution[0]):
                    # threshold the pixel
                    mask[y, x] = 1.0 if img[y, x, 0] == 255 else 0.0
                    if img[y, x, 0] == 255:
                        point = [(y - 127) * img_depth[y, x] / 1.732 / 128, (x - 127) * img_depth[y, x] / 1.732 / 128,
                                 img_depth[y, x]]
                        d_points.append(point)

            trajectory_points = []
            trajectory_robot_points = []
            point = []

            depth_mask = mask * img_depth_0_1
            for y in range(1, len(ffitRound)):
                a = ffitRound[y]
                b = int(round(ffit[y]))
                point = [(a - 127) * img_depth[a, b] / 1.732 / 128, (b - 127) * img_depth[a, b] / 1.732 / 128,
                         img_depth[a, b]]
                trajectory_points.append(point)
                mask = cv2.circle(mask, (int(round(ffit[y])), ffitRound[y]), 2, (0, 0, 255), -1)
                img_rgb = cv2.circle(img_rgb, (int(round(ffit[y])), ffitRound[y]), 2, (0, 0, 255), -1)
                # image_depth = cv2.circle(image_depth, (int(round(ffit[y])), ffitRound[y]), 2, (0, 0, 255), -1)
                depth_mask = cv2.circle(depth_mask, (int(round(ffit[y])), ffitRound[y]), 2, (0, 0, 255), -1)

            trajectory_points_xyz = np.array(trajectory_points)
            trajectory_points_xyz_temp = trajectory_points_xyz
            #trajectory_points_xyz_temp[:][0] = -trajectory_points_xyz[:][1]
            #trajectory_points_xyz_temp[:][1] = -trajectory_points_xyz[:][0]
            #trajectory_points_xyz_temp[:][3] = -trajectory_points_xyz[:][3]

            mat_v0 = np.linalg.inv(mat_v0)

            trajectory_points_XYZ = np.dot(mat_v0, trajectory_points_xyz_temp.T).T
            trajectory_points_XYZ_temp = trajectory_points_XYZ
            #trajectory_points_XYZ_temp[:][0] = trajectory_points_XYZ[:][1]
            #trajectory_points_XYZ_temp[:][1] = trajectory_points_XYZ[:][0]

            trajectory_points_XYZ = trajectory_points_XYZ_temp + np.array(pos_v0)

#             trajectory_points_XYZ[:][0] = trajectory_points_XYZ[:][0] + np.array(pos_v0)[0]
#             trajectory_points_XYZ[:][1] = trajectory_points_XYZ[:][1] + np.array(pos_v0)[1]
#             trajectory_points_XYZ[:][2] = trajectory_points_XYZ[:][2] + np.array(pos_v0)[2]
            if imcount == 1:
                trajectory_points_robot = trajectory_points_xyz
                for k in range(2, len(trajectory_points_XYZ), 10):
                    res, objectHandles = vrep.simxCopyPasteObjects(clientID, [trajectory_sphere_ID],
                                                                   vrep.simx_opmode_blocking)
                    # This line is to use more direct information from camera coordinate system
                    vrep.simxSetObjectPosition(clientID, objectHandles[0], v0,
                                               ([np.float32(-trajectory_points_xyz[k][1]), np.float32(-trajectory_points_xyz[k][0]), np.float32(trajectory_points_xyz[k][2])]),
                                               vrep.simx_opmode_oneshot)
                    time.sleep(0.5)
                    #vrep.simxSetObjectPosition(clientID, objectHandles[0], robot_base_ID,
                    #                           ([np.float32(trajectory_points_XYZ[k][1]), np.float32(trajectory_points_XYZ[k][0]),
                    #                             np.float32(trajectory_points_XYZ[k][2])]),
                    #                            vrep.simx_opmode_oneshot)
            # o3d.visualization.draw_geometries(d_points)
            points = np.random.rand(10000, 3)
            # point_cloud.points
            if imcount == 1:
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(d_points)
                vis.add_geometry(point_cloud)
                point_cloud_trajectory = o3d.geometry.PointCloud()
                point_cloud_trajectory.points = o3d.utility.Vector3dVector(trajectory_points)
                colors = [[0, 0, 0] for i in range(len(trajectory_points))]
                point_cloud_trajectory.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(point_cloud_trajectory)
                trajectory_robot_points = np.array(trajectory_points)
                trajectory_robot_points[:, 2] = trajectory_robot_points[:, 2] - 0.005
                point_cloud_trajectory_robot = o3d.geometry.PointCloud()
                point_cloud_trajectory_robot.points = o3d.utility.Vector3dVector(list(trajectory_robot_points))
                colors_robot = [[0, 1, 0] for i in range(len(trajectory_robot_points))]
                point_cloud_trajectory_robot.colors = o3d.utility.Vector3dVector(colors_robot)
                vis.add_geometry(point_cloud_trajectory_robot)
            # o3d.visualization.draw_geometries([point_cloud])
            point_cloud.points = o3d.utility.Vector3dVector(d_points)
            point_cloud_trajectory.points = o3d.utility.Vector3dVector(trajectory_points)
            colors = [[1, 0, 0] for i in range(len(trajectory_points))]
            point_cloud_trajectory.colors = o3d.utility.Vector3dVector(colors)
            trajectory_robot_points = np.array(trajectory_points)
            trajectory_robot_points[:, 2] = trajectory_robot_points[:, 2] - 0.005
            point_cloud_trajectory_robot.points = o3d.utility.Vector3dVector(list(trajectory_robot_points))
            colors = [[0, 1, 0] for i in range(len(trajectory_robot_points))]
            point_cloud_trajectory_robot.colors = o3d.utility.Vector3dVector(colors)
            if imcount > 1:
                vis.update_geometry(point_cloud)
                vis.update_geometry(point_cloud_trajectory)
                vis.update_geometry(point_cloud_trajectory_robot)
                vis.poll_events()
                vis.update_renderer()
            # cv2.imshow('image', img)
            cv2.imshow('mask', mask)
            cv2.imshow('image_depth', img_depth)
            cv2.imshow('depth_mask', depth_mask)
            cv2.imshow('image_rgb', img_rgb)
            cv2.waitKey(1)

            # Wait for the robot of trajectory planning
            for k in range(2, len(trajectory_points_robot), 1):
                vrep.simxSetObjectPosition(clientID, Robot_target_ID, v0,
                                           ([np.float32(-trajectory_points_robot[k][1]),
                                             np.float32(-trajectory_points_robot[k][0]),
                                             np.float32(trajectory_points_robot[k][2])]),
                                           vrep.simx_opmode_oneshot)
                time.sleep(0.1)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print(res)
    vrep.simxFinish(clientID)
else:
    print('Failed connecting to remote API server')
print('Program ended')