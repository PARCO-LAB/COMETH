from . import rotation
import numpy as np

class ComplementaryFilter:
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.prev_acc = 0
        self.prev_gyro = 0

    def update(self, accel, gyro, dt= 1/60):
        """
        accel: [ax, ay, az] in m/s^2
        gyro: [gx, gy, gz] in rad/s
        dt: timestep in seconds
        """
        ax, ay, az = accel
        gx, gy, gz = gyro

        cr = np.cos(self.roll)
        sr = np.sin(self.roll)

        cp = np.cos(self.pitch)
        sp = np.sin(self.pitch)

        # Gimbal-lock check
        if np.abs(cp)<0.0001:
            if cp > 0 :
                cp = np.cos(np.deg2rad(89.99))
                sp = np.sin(np.deg2rad(89.99))
            else:
                cp = np.cos(np.deg2rad(90.01))
                sp = np.sin(np.deg2rad(90.01))

        # Integrate gyro to get angles
        self.roll += gx * dt + sr*(sp/cp)*gy*dt + cr*(sp/cp)*gz*dt
        self.pitch += cr*gy * dt - sr*gz*dt
        self.yaw += (sr/cp)*gy*dt+(cr/cp)*gz *dt

        # Compute angles from accelerometer
        roll_acc = np.arctan2(ay, az)
        pitch_acc = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

        # Chose  alpha based on accel norm
        if np.abs(np.linalg.norm(accel)-1)>0.2:
            alpha = 1
        elif np.abs(np.linalg.norm(accel)-1)<0.001:
            alpha = 0
        else:
            alpha = self.alpha

        
        # Apply complementary filter
        self.roll = alpha * self.roll + (1 - alpha) * roll_acc
        self.pitch = alpha * self.pitch + (1 - alpha) * pitch_acc

        return self.roll, self.pitch, self.yaw

    def correct(self, T):
        self.roll, self.pitch, self.yaw = rotation.rotmat_to_euler(T[:3,:3])
