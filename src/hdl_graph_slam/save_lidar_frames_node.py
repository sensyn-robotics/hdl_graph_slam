#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
import threading
from queue import Queue
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import struct


class LidarFrameSaver:
    def __init__(self):
        rospy.init_node('save_lidar_frames_node', anonymous=False)

        # Get parameters
        self.output_dir = rospy.get_param('~output_dir', os.path.expanduser('~/lidar_frames'))

        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            rospy.loginfo("Created output directory: %s", self.output_dir)

        # Async saving queue
        self.save_queue = Queue()
        self.save_worker_running = True
        self.save_worker_thread = threading.Thread(target=self.save_worker)
        self.save_worker_thread.start()

        # Frame counter
        self.frame_count = 0

        # Subscribe to point cloud topic
        self.sub = rospy.Subscriber('/os_cloud_node/points', PointCloud2, self.cloud_callback, queue_size=10)

        rospy.loginfo("LiDAR frame saver started. Saving to: %s", self.output_dir)

    def cloud_callback(self, msg):
        """Callback for point cloud messages"""
        # Add to save queue (non-blocking)
        self.save_queue.put(msg)

    def save_worker(self):
        """Worker thread for async file saving"""
        rospy.loginfo("Save worker thread started")

        while self.save_worker_running or not self.save_queue.empty():
            try:
                # Get message from queue with timeout
                msg = self.save_queue.get(timeout=0.1)
                self.save_point_cloud(msg)
                self.save_queue.task_done()
            except:
                continue

        rospy.loginfo("Save worker thread finished")

    def save_point_cloud(self, msg):
        """Save point cloud to PCD file"""
        # Create filename from timestamp
        timestamp_sec = msg.header.stamp.secs
        timestamp_nsec = msg.header.stamp.nsecs
        filename = "{:d}_{:09d}.pcd".format(timestamp_sec, timestamp_nsec)
        filepath = os.path.join(self.output_dir, filename)

        try:
            # Convert PointCloud2 to list of points
            points = []
            for p in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                points.append(p)

            # Write PCD file (binary format)
            with open(filepath, 'wb') as f:
                # Write PCD header
                f.write(b"# .PCD v0.7 - Point Cloud Data file format\n")
                f.write(b"VERSION 0.7\n")
                f.write(b"FIELDS x y z intensity\n")
                f.write(b"SIZE 4 4 4 4\n")
                f.write(b"TYPE F F F F\n")
                f.write(b"COUNT 1 1 1 1\n")
                f.write(("WIDTH {:d}\n".format(len(points))).encode())
                f.write(b"HEIGHT 1\n")
                f.write(b"VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(("POINTS {:d}\n".format(len(points))).encode())
                f.write(b"DATA binary\n")

                # Write point data
                for p in points:
                    f.write(struct.pack('ffff', p[0], p[1], p[2], p[3]))

            self.frame_count += 1

            # Log progress every 100 frames
            if self.frame_count % 100 == 0:
                rospy.loginfo("Saved %d frames, queue size: %d", self.frame_count, self.save_queue.qsize())

        except Exception as e:
            rospy.logwarn("Failed to save PCD file %s: %s", filepath, str(e))

    def shutdown(self):
        """Shutdown handler"""
        rospy.loginfo("Shutting down LiDAR frame saver...")

        # Stop worker thread
        self.save_worker_running = False
        self.save_worker_thread.join()

        rospy.loginfo("Total frames saved: %d", self.frame_count)


def main():
    try:
        saver = LidarFrameSaver()
        rospy.on_shutdown(saver.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()