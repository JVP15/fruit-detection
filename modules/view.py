import PySimpleGUI as sg
import cv2


class View:
    menu_def = [['&File', ['Open Video File', 'Open Camera', '---', 'E&xit']],
                ['&View', ['!Confidence View', 'Harvestability View']]
                ]
    width = 800
    height = 800

    def __init__(self):
        sg.theme('light green')
        self.quit = False
        self.has_image = False
        self.window = None
        self.source = None
        self.display = 'confidence'

    def start(self):
        video_file_button = sg.Button('Open Video File', visible=False, key='--VIDEO-FILE--',  font=('Any', 28))
        camera_button = sg.Button('Open Camera', visible=False, key='--CAMERA--',  font=('Any', 28))
        layout = [
            [sg.VPush()],
            [sg.Push(), sg.Text('Loading Models...', font=('Any', 28), key='--MAIN-TEXT--'),
                sg.Image(key='--IMAGE--'), video_file_button, camera_button, sg.Push()],
            [sg.VPush()],
            [sg.Menu(self.menu_def, tearoff=False, pad=(200, 1), key='--MENU--')],
        ]
        self.window = sg.Window('Fruit Detection', layout, size=(self.width, self.height))
        _, _ = self.window.read(timeout=13)

    def update_image(self, img):
        imgbytes = cv2.imencode('.png', img)[1].tobytes()

        if not self.has_image:
            self.has_image = True
            self.window['--MAIN-TEXT--'].update(visible=False)

        self.window['--IMAGE--'].update(data=imgbytes)

    def update_source(self, source):
        self.source = source
        self.window['--MAIN-TEXT--'].update(visible=False)

        if source is None:
            self.window['--VIDEO-FILE--'].update(visible=True)
            self.window['--CAMERA--'].update(visible=True)
            self.window['--IMAGE--'].update(data=None)
        else:
            self.window['--VIDEO-FILE--'].update(visible=False)
            self.window['--CAMERA--'].update(visible=False)

    def display_source_error(self, source):
        sg.popup_error('Could not open source: ' + str(source))

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

        elif events == 'Open Video File' or events == '--VIDEO-FILE--':
            file_source = sg.popup_get_file('Select video file:', no_window=True)

            if file_source is not None and file_source != '':
                self.update_source(file_source)

        elif events == 'Open Camera' or events == '--CAMERA--':
            camera_source = sg.popup_get_text('Enter camera index')
            if camera_source is not None:
                try:
                    camera_source = int(camera_source)
                    self.update_source(camera_source)
                except ValueError:
                    sg.popup_error('Camera index must be an integer')

        elif events == 'Exit' or events is None:
            self.close()
