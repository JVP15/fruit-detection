import PySimpleGUI as sg
import cv2

class View:
    def __init__(self):
        sg.theme('light green')
        self.started = False
        self.has_image = False
        self.window = None
        self.display = 'confidence'

    def update(self, img):
        imgbytes = cv2.imencode('.png', img)[1].tobytes()

        if not self.has_image:
            self.has_image = True
            layout = [
                [sg.Button('Confidence View', size=(20, 1), font=('Any', 14)),
                 sg.Button('Harvestability View', size=(20, 1), font=('Any', 14))],
                [sg.Image(data=imgbytes, key='_IMAGE_')]
            ]
            # we have to create a new window to update the layout, so we close the old one first to make way for the new one
            self.window.close()
            self.window = sg.Window('Fruit Detection', layout, finalize=True)
            self.window['Confidence View'].update(disabled=True)

        image_elem = self.window['_IMAGE_']
        image_elem.update(data=imgbytes)

    def start(self):
        self.started = True
        layout = [
            [sg.Text('Loading Models...', font=('Any', 14))],
        ]
        self.window = sg.Window('Fruit Detection', layout, size=(500, 500))
        _ = self.window.read(timeout=0)

    def close(self):
        self.window.close()

    def get_event(self):
        return self.window.read(timeout=0)

    def get_display(self):
        return self.display

    def set_display(self, display):
        if display == 'Confidence View':
            self.window['Confidence View'].update(disabled=True)
            self.window['Harvestability View'].update(disabled=False)
            self.display = 'confidence'
        elif display == 'Harvestability View':
            self.window['Confidence View'].update(disabled=False)
            self.window['Harvestability View'].update(disabled=True)
            self.display = 'harvestability'