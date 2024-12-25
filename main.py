import math
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import wx

startframe = None
testframe = None

class StartFrame(wx.Frame):
    global testframe

    def __init__(self, *args, **kw):
        super(StartFrame, self).__init__(*args, **kw)

        self.panel = wx.Panel(self)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        toptext = wx.StaticText(self.panel, label="Pick a video file and select your shooting hand")
        font = toptext.GetFont()
        font.PointSize += 10
        font = font.Bold()
        toptext.SetFont(font)
        self.sizer.Add(toptext, 0, wx.ALL, 10)

        filetext = wx.StaticText(self.panel, label="Select a video file:")
        self.sizer.Add(filetext, 0, wx.ALL, 10)

        self.filepicker = wx.FilePickerCtrl(self.panel)
        self.sizer.Add(self.filepicker, 0, wx.ALL, 10)

        radiotext = wx.StaticText(self.panel, label="Select your shooting hand:")
        self.sizer.Add(radiotext, 0, wx.ALL, 10)

        self.radioright = wx.RadioButton(self.panel, label="Right", style=wx.RB_GROUP)
        self.radioleft = wx.RadioButton(self.panel, label="Left")
        self.sizer.Add(self.radioright, 0, wx.ALL, 10)
        self.sizer.Add(self.radioleft, 0, wx.ALL, 10)

        self.warningtext = wx.StaticText(self.panel, label="")
        self.warningtext.SetForegroundColour(wx.Colour(255, 0, 0))
        self.sizer.Add(self.warningtext, 0, wx.ALL, 10)

        processbutton = wx.Button(self.panel, label="Process!")
        processbutton.Bind(wx.EVT_BUTTON, self.onProcess)
        self.sizer.Add(processbutton, 0, wx.ALL, 10)

        self.panel.SetSizer(self.sizer)

        self.SetSize((800, 400))

        self.Bind(wx.EVT_CLOSE, self.onClose)

    def onProcess(self, event):
        input_filename = self.filepicker.GetPath()
        if input_filename == "" or not input_filename.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
            self.warningtext.SetLabel("Invalid video file!")
            return
        else:
            self.warningtext.SetLabel("")

        print("Using file:" + input_filename)

        loadingdialog = wx.ProgressDialog("Free Throw Trainer - Processing...", "Analyzing your video...",
                                          parent=self, style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_ELAPSED_TIME |
                                                             wx.PD_REMAINING_TIME)
        loadingdialog.Show()

        # ACTUAL PROCESSING

        video_read = cv2.VideoCapture(input_filename)

        hand = "right" if self.radioright.GetValue() else "left"

        model_path = "pose_landmarker_heavy.task"

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a pose landmarker instance with the video mode:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO)

        detector = PoseLandmarker.create_from_options(options)

        video_width = int(video_read.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_read.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_read.get(cv2.CAP_PROP_FPS)
        frametime_ms = 1000 / fps
        timestamp_ms = 0.0
        frame_num = 0

        loadingdialog.SetRange(total_frames)

        video_write = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (video_width, video_height))

        # Plot data
        plot_x = np.linspace(0, total_frames * frametime_ms, total_frames)
        plot_elbow_y = np.empty(total_frames)
        plot_knees_y = np.empty(total_frames)
        plot_shoulder_y = np.empty(total_frames)

        def get_angle3d(a: (float, float, float), c: (float, float, float), b: (float, float, float)) -> float:
            # Law of cosines: c^2 = a^2 + b^2 - 2ab * cos(C)
            # C = acos((a^2 + b^2 - c^2)/(2ab))
            a_side_sqr = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2 + (a[2] - c[2]) ** 2
            b_side_sqr = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2 + (b[2] - c[2]) ** 2
            c_side_sqr = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
            a_side = math.sqrt(a_side_sqr)
            b_side = math.sqrt(b_side_sqr)

            return math.acos((a_side_sqr + b_side_sqr - c_side_sqr) / (2 * a_side * b_side)) * (180 / math.pi)

        def get_point3d(i: int) -> (float, float, float):
            return pose3d[i].x, pose3d[i].y, pose3d[i].z

        def get_point(i: int) -> (int, int):
            return round(pose[i].x * video_width), round(pose[i].y * video_height)

        print("Processing video...")
        while video_read.isOpened():
            ret, frame = video_read.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            pose_landmarker_result = detector.detect_for_video(mp_frame, round(timestamp_ms))
            for i in range(len(pose_landmarker_result.pose_landmarks)):
                pose_landmarks = pose_landmarker_result.pose_landmarks[i]
                for j, landmark in enumerate(pose_landmarks):
                    point = (round(landmark.x * video_width), round(landmark.y * video_height))
                    cv2.circle(frame, point, 5, (255, 0, 0), -1)

            pose3d = pose_landmarker_result.pose_world_landmarks[0]

            pose = pose_landmarker_result.pose_landmarks[0]

            # Right elbow
            right_shoulder3d = get_point3d(12)
            right_elbow3d = get_point3d(14)
            right_wrist3d = get_point3d(16)
            right_shoulder = get_point(12)
            right_elbow = get_point(14)
            right_wrist = get_point(16)
            right_elbow_angle = get_angle3d(right_shoulder3d, right_elbow3d, right_wrist3d)

            if hand == "right":
                cv2.line(frame, right_shoulder, right_elbow, (0, 255, 0), 3)
                cv2.line(frame, right_elbow, right_wrist, (0, 255, 0), 3)

                cv2.putText(frame, f"Elbow angle (right): {round(right_elbow_angle)}", (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Left elbow
            left_shoulder3d = get_point3d(11)
            left_elbow3d = get_point3d(13)
            left_wrist3d = get_point3d(15)
            left_shoulder = get_point(11)
            left_elbow = get_point(13)
            left_wrist = get_point(15)
            left_elbow_angle = get_angle3d(left_shoulder3d, left_elbow3d, left_wrist3d)

            if hand == "left":
                cv2.line(frame, left_shoulder, left_elbow, (0, 255, 0), 3)
                cv2.line(frame, left_elbow, left_wrist, (0, 255, 0), 3)

                cv2.putText(frame, f"Elbow angle (left): {round(left_elbow_angle)}", (10, 30), cv2.FONT_HERSHEY_DUPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                # Right knee
            right_hip3d = get_point3d(24)
            right_knee3d = get_point3d(26)
            right_ankle3d = get_point3d(28)
            right_hip = get_point(24)
            right_knee = get_point(26)
            right_ankle = get_point(28)
            right_knee_angle = get_angle3d(right_hip3d, right_knee3d, right_ankle3d)

            cv2.line(frame, right_hip, right_knee, (0, 255, 0), 3)
            cv2.line(frame, right_knee, right_ankle, (0, 255, 0), 3)

            # Left knee
            left_hip3d = get_point3d(23)
            left_knee3d = get_point3d(25)
            left_ankle3d = get_point3d(27)
            left_hip = get_point(23)
            left_knee = get_point(25)
            left_ankle = get_point(27)
            left_knee_angle = get_angle3d(left_hip3d, left_knee3d, left_ankle3d)

            cv2.line(frame, left_hip, left_knee, (0, 255, 0), 3)
            cv2.line(frame, left_knee, left_ankle, (0, 255, 0), 3)

            # Average knees angle
            knees_angle = (right_knee_angle + left_knee_angle) / 2

            cv2.putText(frame, f"Average knees angle: {round(knees_angle)}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)

            # Right shoulder
            right_shoulder_angle = get_angle3d(right_hip3d, right_shoulder3d, right_elbow3d)

            if hand == "right":
                cv2.line(frame, right_hip, right_shoulder, (0, 255, 0), 3)

                cv2.putText(frame, f"Shoulder angle (right): {round(right_shoulder_angle)}", (10, 90),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Left shoulder
            left_shoulder_angle = get_angle3d(left_hip3d, left_shoulder3d, left_elbow3d)

            if hand == "left":
                cv2.line(frame, left_hip, left_shoulder, (0, 255, 0), 3)

                cv2.putText(frame, f"Shoulder angle (left): {round(left_shoulder_angle)}", (10, 90),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Time
            cv2.putText(frame, f"Time: {round(timestamp_ms)} ms", (10, 120), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_write.write(frame)
            timestamp_ms += frametime_ms

            if hand == "right":
                plot_elbow_y[frame_num] = right_elbow_angle
            elif hand == "left":
                plot_elbow_y[frame_num] = left_elbow_angle

            plot_knees_y[frame_num] = knees_angle

            if hand == "right":
                plot_shoulder_y[frame_num] = right_shoulder_angle
            elif hand == "left":
                plot_shoulder_y[frame_num] = left_shoulder_angle

            frame_num += 1
            loadingdialog.Update(frame_num)

        print("Video processing complete!")

        video_read.release()
        video_write.release()

        if hand == "right":
            plt.plot(plot_x, plot_elbow_y, label="elbow angle (right)")
        elif hand == "left":
            plt.plot(plot_x, plot_elbow_y, label="elbow angle (left)")

        plt.plot(plot_x, plot_knees_y, label="knees angle")

        if hand == "right":
            plt.plot(plot_x, plot_shoulder_y, label="shoulder angle (right)")
        elif hand == "left":
            plt.plot(plot_x, plot_shoulder_y, label="shoulder angle (left)")

        plt.xlabel("Time (milliseconds)")
        plt.ylabel("Angle (degrees)")
        plt.xticks(np.arange(0.0, (total_frames + 1.0) * frametime_ms, 200.0))
        plt.legend()
        plt.savefig("output_plot.png", dpi=200)
        plt.close()

        # END ACTUAL PROCESSING

        loadingdialog.Destroy()

        testframe.Show()
        self.Hide()

    def onClose(self, event):
        testframe.Destroy()
        self.Destroy()

class TestFrame(wx.Frame):
    panel = None
    sizer = None
    comparison_choice = None
    comparison_image = None
    output_bottom_text = None
    comparison_bottom_text = None
    class Sample:
        def __init__(self, name, title):
            self.plot_filename = "samples/" + name + "_plot.png"
            self.wximage = wx.Image(self.plot_filename).ConvertToBitmap()
            self.video_filename = "samples/" + name + ".mp4"
            self.title = title
    samples = None

    def __init__(self, *args, **kw):
        super(TestFrame, self).__init__(*args, **kw)

        self.Bind(wx.EVT_CLOSE, self.onClose)
        self.Bind(wx.EVT_SHOW, self.onShow)

    def onShow(self, event):
        self.samples = {
            "curry1": self.Sample("curry1", "Stephen Curry (Career 91.0 FT%)"),
            "nash1": self.Sample("nash1", "Steve Nash (Career 90.4 FT%)"),
            "lillard1": self.Sample("lillard1", "Damian Lillard (Career 89.8 FT%)"),
            "irving1": self.Sample("irving1", "Kyrie Irving (Career 88.6 FT%)")
        }

        self.panel = wx.Panel(self)

        self.sizer = wx.FlexGridSizer(4, 2, (10, 10))

        self.sizer.SetFlexibleDirection(wx.VERTICAL)

        self.sizer.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_ALL)
        self.sizer.AddGrowableCol(0, 0)
        self.sizer.AddGrowableCol(1, 0)
        # self.sizer.AddGrowableRow(0, 0)
        self.sizer.AddGrowableRow(2, 0)

        output_text = wx.StaticText(self.panel, label="Your Results:")
        font = output_text.GetFont()
        font.PointSize += 10
        font = font.Bold()
        output_text.SetFont(font)
        self.sizer.Add(output_text, 0, wx.ALL | wx.EXPAND, 10)

        comparison_text = wx.StaticText(self.panel, label="Compare...")
        font = comparison_text.GetFont()
        font.PointSize += 10
        font = font.Bold()
        comparison_text.SetFont(font)
        self.sizer.Add(comparison_text, 0, wx.ALL | wx.EXPAND, 10)

        empty_text = wx.StaticText(self.panel, label="")
        self.sizer.Add(empty_text, 0, wx.ALL | wx.EXPAND, 10)

        self.comparison_choice = wx.Choice(self.panel, choices=[self.samples["curry1"].title, self.samples["nash1"].title, self.samples["lillard1"].title, self.samples["irving1"].title])
        self.comparison_choice.Bind(wx.EVT_CHOICE, self.onChoose)
        self.sizer.Add(self.comparison_choice, 0, wx.ALL | wx.EXPAND, 10)

        output_plot_image = wx.GenericStaticBitmap(self.panel, bitmap=wx.Image("output_plot.png").ConvertToBitmap())
        output_plot_image.SetScaleMode(wx.GenericStaticBitmap.Scale_AspectFit)
        output_plot_image.SetMinSize((100, 100))
        self.sizer.Add(output_plot_image, 0, wx.ALL | wx.SHAPED, 10)

        self.comparison_image = wx.GenericStaticBitmap(self.panel, bitmap=self.samples["nash1"].wximage)
        self.comparison_image.SetScaleMode(wx.GenericStaticBitmap.Scale_AspectFit)
        self.comparison_image.SetMinSize((100, 100))
        self.sizer.Add(self.comparison_image, 0, wx.ALL | wx.SHAPED, 10)

        self.output_bottom_text = wx.StaticText(self.panel, label="An annotated video is available at output.mp4")
        self.sizer.Add(self.output_bottom_text, 0, wx.ALL | wx.EXPAND, 10)

        self.comparison_bottom_text = wx.StaticText(self.panel, label="An annotated video is available at " + self.samples["nash1"].video_filename)
        self.sizer.Add(self.comparison_bottom_text, 0, wx.ALL | wx.EXPAND, 10)

        self.panel.SetSizer(self.sizer)

        self.SetSize(1200, 800)

    def onChoose(self, event):
        choice = self.comparison_choice.GetString(self.comparison_choice.GetSelection())
        if choice == self.samples["curry1"].title:
            self.comparison_image.SetBitmap(self.samples["curry1"].wximage)
            self.comparison_bottom_text.SetLabel("An annotated video is available at " + self.samples["curry1"].video_filename)
        elif choice == self.samples["nash1"].title:
            self.comparison_image.SetBitmap(self.samples["nash1"].wximage)
            self.comparison_bottom_text.SetLabel("An annotated video is available at " + self.samples["nash1"].video_filename)
        elif choice == self.samples["lillard1"].title:
            self.comparison_image.SetBitmap(self.samples["lillard1"].wximage)
            self.comparison_bottom_text.SetLabel("An annotated video is available at " + self.samples["lillard1"].video_filename)
        elif choice == self.samples["irving1"].title:
            self.comparison_image.SetBitmap(self.samples["irving1"].wximage)
            self.comparison_bottom_text.SetLabel("An annotated video is available at " + self.samples["irving1"].video_filename)
        self.sizer.Layout()

    def onClose(self, event):
        startframe.Destroy()
        self.Destroy()

if __name__ == '__main__':
    app = wx.App()
    startframe = StartFrame(None, title="Free Throw Trainer")
    testframe = TestFrame(None, title="Free Throw Trainer - Results")
    startframe.Show()
    app.MainLoop()
