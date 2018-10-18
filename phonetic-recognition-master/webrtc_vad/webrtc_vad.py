import webrtcvad
import collections
import sys
import signal
import pyaudio

from array import array
from struct import pack
import wave
import time
import speech_rnn.speech1 as speech1

# WAV_FILES = ["recording.wav"]
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
CHUNK_DURATION_MS = 30       # 参数 10, 20 and 30 (ms)
PADDING_DURATION_MS = 1500   # 1s判断
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # 读入chunk
CHUNK_BYTES = CHUNK_SIZE * 2  # 16bit = 2 bytes, PCM
NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)  # 400 ms/ 30ms
NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2

START_OFFSET = int(NUM_WINDOW_CHUNKS * CHUNK_DURATION_MS * 0.5 * RATE)

vad = webrtcvad.Vad(1)

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 start=False,
                 # input_device_index=2,
                 frames_per_buffer=CHUNK_SIZE)


got_a_sentence = False
leave = False


def handle_int(sig, chunk):
    global leave, got_a_sentence
    leave = True
    got_a_sentence = True

#从麦克风记录并输出结果数据到“path”
def record_to_file(path, data, sample_width):
    # sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

#平均输出音量
def normalize(snd_data):
    MAXIMUM = 32767  # 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r
#如果发生SIGINT信号则中断，转去执行handle_int()函数
signal.signal(signal.SIGINT, handle_int)

while not leave:
    ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
    triggered = False
    voiced_frames = []
    ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
    ring_buffer_index = 0

    ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
    ring_buffer_index_end = 0
    buffer_in = ''
    # WangS
    raw_data = array('h')
    index = 0
    start_point = 0
    StartTime = time.time()
    print("* recording: ")
    stream.start_stream()

    while not got_a_sentence and not leave:
        chunk = stream.read(CHUNK_SIZE)
        # add WangS
        raw_data.extend(array('h', chunk))
        index += CHUNK_SIZE
        TimeUse = time.time() - StartTime

        active = vad.is_speech(chunk, RATE)

        sys.stdout.write('1' if active else '_')
        ring_buffer_flags[ring_buffer_index] = 1 if active else 0
        ring_buffer_index += 1
        ring_buffer_index %= NUM_WINDOW_CHUNKS

        ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
        ring_buffer_index_end += 1
        ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END

        # start point detection
        if not triggered:
            ring_buffer.append(chunk)
            num_voiced = sum(ring_buffer_flags)
            NUM_WINDOW_CHUNKS08 = 0.8 * NUM_WINDOW_CHUNKS
            if num_voiced > 0.8 * NUM_WINDOW_CHUNKS:
                sys.stdout.write(' Open ')
                triggered = True
                start_point = index - CHUNK_SIZE * 20  # start point
                # voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        # end point detection
        else:
            # voiced_frames.append(chunk)
            ring_buffer.append(chunk)
            num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)
            if num_unvoiced > 0.90 * NUM_WINDOW_CHUNKS_END or TimeUse > 10:
                sys.stdout.write(' Close ')
                triggered = False
                got_a_sentence = True

        # sys.stdout.flush()

    sys.stdout.write('\n')
    # data = b''.join(voiced_frames)

    stream.stop_stream()
    print("* done recording")
    beginTime = time.time()
    got_a_sentence = False

    # write to file
    raw_data.reverse()
    for index in range(start_point):
        raw_data.pop()
    raw_data.reverse()
    raw_data = normalize(raw_data)
    record_to_file("recording.wav", raw_data, 2)

    leave = True


stream.close()

#调用模型识别音频文件
# speech1.predict("recording.wav")
# runTime = time.time() - beginTime
# print("runTime: ", runTime)