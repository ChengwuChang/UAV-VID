"""
PyTeapot module for drawing rotating cube using OpenGL as per
quaternion or yaw, pitch, roll angles received over serial port.
"""

import pygame
import math
import pandas as pd
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

class PyTeapot():
    def __init__(self):
        super().__init__()
        self.useSerial = False  # set true for using serial for data transmission, false for wifi
        self.useQuat = False  # set true for using quaternions, false for using y,p,r angles
        self.main()

    def main(self):
        video_flags = OPENGL | DOUBLEBUF
        pygame.init()
        screen = pygame.display.set_mode((640, 480), video_flags)
        pygame.display.set_caption("GY-521 orientation visualization")
        self.resizewin(640, 480)
        self.init()
        frames = 0
        ticks = pygame.time.get_ticks()

        clock = pygame.time.Clock()

        with open(r"C:\Users\ASUS\Downloads\aircraft_attitude_2024-04-19_21-26-16.csv", 'r') as file:
            reader = pd.read_csv(file)
            num_rows = len(reader)
            row_index = 0
            running = True

            while running:
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        pygame.quit()
                        running = False
                        break
                if not running:
                    break

                if row_index < num_rows:
                    # yaw_deg = reader.at[row_index, 'Accel X']
                    yaw_deg = 0
                    pitch_deg = reader.at[row_index, 'Pitch (degrees)']
                    roll_deg = reader.at[row_index, 'Roll (degrees)']

                    # yaw_rad = math.radians(yaw_deg)
                    # pitch_rad = math.radians(pitch_deg)
                    # roll_rad = math.radians(roll_deg)

                    print(yaw_deg, pitch_deg, roll_deg)

                    self.draw(1, yaw_deg, pitch_deg, roll_deg)
                    row_index += 1
                    ticks = pygame.time.get_ticks()  # 更新上一時間點

                pygame.display.flip()
                frames += 1
                clock.tick(20)

        print("fps: %d" % ((frames * 1000) / (pygame.time.get_ticks() - ticks)))

    def resizewin(self, width, height):
        """
        For resizing window
        """
        if height == 0:
            height = 1
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1.0 * width / height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def init(self):
        glShadeModel(GL_SMOOTH)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

    def draw(self, w, nx, ny, nz):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, 0.0, -7.0)

        self.drawText((-2.6, 1.8, 2), "CCU UAV", 18)
        self.drawText((-2.6, 1.6, 2), "Visualize Euler angles data", 16)
        self.drawText((-2.6, -2, 2), "Press Escape to exit.", 16)

        if (self.useQuat):
            [yaw, pitch, roll] = self.quat_to_ypr([w, nx, ny, nz])
            self.drawText((-2.6, -1.8, 2), "Yaw: %f, Pitch: %f, Roll: %f" % (yaw, pitch, roll), 16)
            glRotatef(2 * math.acos(w) * 180.00 / math.pi, -1 * nx, nz, ny)
        else:
            yaw = nx
            pitch = ny
            roll = nz
            self.drawText((-2.6, -1.8, 2), "Yaw: %f, Pitch: %f, Roll: %f" % (yaw, pitch, roll), 16)
            glRotatef(-roll, 0.00, 0.00, 1.00)
            glRotatef(pitch, 1.00, 0.00, 0.00)
            glRotatef(yaw, 0.00, 1.00, 0.00)

        glBegin(GL_QUADS)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(1.0, 0.2, -1.0)
        glVertex3f(-1.0, 0.2, -1.0)
        glVertex3f(-1.0, 0.2, 1.0)
        glVertex3f(1.0, 0.2, 1.0)

        glColor3f(1.0, 0.5, 0.0)
        glVertex3f(1.0, -0.2, 1.0)
        glVertex3f(-1.0, -0.2, 1.0)
        glVertex3f(-1.0, -0.2, -1.0)
        glVertex3f(1.0, -0.2, -1.0)

        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(1.0, 0.2, 1.0)
        glVertex3f(-1.0, 0.2, 1.0)
        glVertex3f(-1.0, -0.2, 1.0)
        glVertex3f(1.0, -0.2, 1.0)

        glColor3f(1.0, 1.0, 0.0)
        glVertex3f(1.0, -0.2, -1.0)
        glVertex3f(-1.0, -0.2, -1.0)
        glVertex3f(-1.0, 0.2, -1.0)
        glVertex3f(1.0, 0.2, -1.0)

        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(-1.0, 0.2, 1.0)
        glVertex3f(-1.0, 0.2, -1.0)
        glVertex3f(-1.0, -0.2, -1.0)
        glVertex3f(-1.0, -0.2, 1.0)

        glColor3f(1.0, 0.0, 1.0)
        glVertex3f(1.0, 0.2, -1.0)
        glVertex3f(1.0, 0.2, 1.0)
        glVertex3f(1.0, -0.2, 1.0)
        glVertex3f(1.0, -0.2, -1.0)
        glEnd()

        glLineWidth(2.0)
        glBegin(GL_LINES)

        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(-1.5, 0.0, 0.0)
        glVertex3f(1.5, 0.0, 0.0)

        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(0.0, -1.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)

        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(0.0, 0.0, -1.6)
        glVertex3f(0.0, 0.0, 1.6)
        glEnd()

        glBegin(GL_TRIANGLES)
        # x-axis
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(1.5, 0.0, 0.0)
        glVertex3f(1.4, 0.1, 0.0)
        glVertex3f(1.4, -0.1, 0.0)
        # y-axis
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(0.0, 1.0, 0.0)
        glVertex3f(0.1, 0.9, 0.0)
        glVertex3f(-0.1, 0.9, 0.0)
        # z-axis
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(0.0, 0.0, 1.6)
        glVertex3f(0.0, 0.1, 1.5)
        glVertex3f(0.0, -0.1, 1.5)
        glEnd()

    def drawText(self, position, textString, size):
        font = pygame.font.SysFont("Courier", size, True)
        textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
        textData = pygame.image.tostring(textSurface, "RGBA", True)
        glRasterPos3d(*position)
        glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

    def quat_to_ypr(self, q):
        yaw = math.atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
        pitch = -math.asin(2.0 * (q[1] * q[3] - q[0] * q[2]))
        roll = math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])
        pitch *= 180.0 / math.pi
        yaw *= 180.0 / math.pi
        yaw -= -0.13  # Declination at Chandrapur, Maharashtra is - 0 degress 13 min
        roll *= 180.0 / math.pi
        return [yaw, pitch, roll]


# if __name__ == '__main__':
#     PyTeapot()
