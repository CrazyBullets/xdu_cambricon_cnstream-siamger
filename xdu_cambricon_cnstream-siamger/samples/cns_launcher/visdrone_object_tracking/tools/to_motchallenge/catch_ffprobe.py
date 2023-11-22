import subprocess
import json
import argparse 
from pathlib import Path

class FFprobe():
    def __init__(self):
        self.filepath = ''
        self._video_info = {}


    def parse(self,filepath):
        self.filepath = filepath
        try:
            res = subprocess.check_output(['ffprobe','-i',self.filepath,'-print_format','json','-show_format','-show_streams','-v','quiet'])
            res = res.decode('utf8')
            self._video_info = json.loads(res)
            # print('_video_info ',self._video_info)
        except Exception as e:
            print(e)
            raise Exception('parse failed')

    def video_width(self):
        streams = self._video_info['streams'][0]
        return (streams['width'])

    def video_height(self):
        streams = self._video_info['streams'][0]
        return (streams['height'])

    def video_name(self):
        fullname = Path(self.filepath)
        return fullname.stem

    def video_seq_length(self):
        return int(self._video_info['streams'][0]['nb_frames'])
    
    def video_frame_rate(self):
        return int(self._video_info['streams'][0]['r_frame_rate'].split('/')[0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse seqinfo by ffprobe')

    parser.add_argument("--filepath", dest = 'filepath', help =
                        "path of result files",
                        default = '', type = str)

    args = parser.parse_args()

    ffprobe = FFprobe()
    ffprobe.parse(args.filepath)

    seq_file = "seqinfo.ini"

    with open(seq_file, 'w') as f:
        f.write("[Sequence]\n")
        f.write("name={}\n".format(ffprobe.video_name()))
        f.write("imDir={}\n".format(ffprobe.video_name()))
        f.write("frameRate={}\n".format(ffprobe.video_frame_rate()))
        f.write("seqLength={}\n".format(ffprobe.video_seq_length()))
        f.write("imWidth={}\n".format(ffprobe.video_width()))
        f.write("imHeight={}\n".format(ffprobe.video_height()))
        f.write("imExt=.jpg\n")
        