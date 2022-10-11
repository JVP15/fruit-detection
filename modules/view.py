import PySimpleGUI as sg
import cv2



class View:
    menu_def = [['&File', ['Open Video File', 'Open Camera', 'E&xit']],
                ['&View', ['!Confidence View', 'Harvestability View']]
                ]

    def __init__(self):
        sg.theme('light green')
        self.quit = False
        self.has_image = False
        self.window = None
        self.display = 'confidence'

    def update(self, img):
        imgbytes = cv2.imencode('.png', img)[1].tobytes()

        if not self.has_image:
            self.has_image = True
            layout = [
                [sg.Image(data=imgbytes, key='--IMAGE--')],
                [sg.Menu(self.menu_def, tearoff=False, pad=(200, 1), key='--MENU--')],
            ]

            # we have to create a new window to update the layout, so we close the old one first to make way for the new one
            self.window.close()
            self.window = sg.Window('Fruit Detection', layout, finalize=True, force_toplevel=True)

        image_elem = self.window['--IMAGE--']
        image_elem.update(data=imgbytes)

    def start(self):
        self.started = True
        layout = [
            [sg.VPush()],
            [sg.Push(), sg.Text('Loading Models...', font=('Any', 28)), sg.Push()],
            [sg.VPush()],
            [sg.Menu(self.menu_def, tearoff=False, pad=(200, 1), key='--MENU--')],
        ]
        self.window = sg.Window('Fruit Detection', layout, size=(400, 400))
        _ = self.window.read(timeout=0)

    def close(self):
        self.quit = True
        self.window.close()

    def get_event(self):
        return self.window.read(timeout=13)

    def get_display(self):
        return self.display

    def process_events(self, events, values):
        if events == 'Confidence View':
            self.display = 'confidence'
            self.menu_def[1][1][0] = '!Confidence View'
            self.menu_def[1][1][1] = 'Harvestability View'
            self.window['--MENU--'].update(self.menu_def)
        elif events == 'Harvestability View':
            self.display = 'harvestability'
            self.menu_def[1][1][0] = 'Confidence View'
            self.menu_def[1][1][1] = '!Harvestability View'
            self.window['--MENU--'].update(self.menu_def)

        elif events == 'Exit' or None:
            self.close()
        # TODO: handle camera and video file events