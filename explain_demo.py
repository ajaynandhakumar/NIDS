import pandas as pd
import numpy as np
from capture import simulate_packet
from preprocess import preprocess_packet, packet_to_dataframe
from detector import NIDSDetector

detector = NIDSDetector()

print('=' * 70)
print('  ATTACK vs NORMAL — உள்ளே என்ன நடக்குது?')
print('=' * 70)

normal_packets = [
    {'src_ip':'192.168.1.5',  'dst_ip':'8.8.8.8',      'protocol':'TCP',  'packet_size':400,  'src_port':52341, 'dst_port':80,   'duration':2.5,   'packet_count':8},
    {'src_ip':'192.168.1.10', 'dst_ip':'1.1.1.1',      'protocol':'UDP',  'packet_size':256,  'src_port':60000, 'dst_port':53,   'duration':0.8,   'packet_count':3},
    {'src_ip':'192.168.1.3',  'dst_ip':'142.250.1.1',  'protocol':'TCP',  'packet_size':620,  'src_port':54200, 'dst_port':443,  'duration':4.0,   'packet_count':12},
]

attack_packets = [
    {'src_ip':'203.0.113.9',  'dst_ip':'192.168.1.1',  'protocol':'ICMP', 'packet_size':1490, 'src_port':31337, 'dst_port':4444, 'duration':0.002, 'packet_count':8000},
    {'src_ip':'45.33.32.156', 'dst_ip':'192.168.1.5',  'protocol':'TCP',  'packet_size':1450, 'src_port':65000, 'dst_port':3389, 'duration':0.001, 'packet_count':5000},
    {'src_ip':'198.51.100.5', 'dst_ip':'192.168.1.2',  'protocol':'TCP',  'packet_size':64,   'src_port':12345, 'dst_port':445,  'duration':0.005, 'packet_count':9999},
]

print()
print('--- [1] NORMAL TRAFFIC (safe packets) ---')
for pkt in normal_packets:
    result = detector.predict(pkt)
    br = (pkt['packet_size'] * pkt['packet_count']) / (pkt['duration'] + 1e-6)
    src = pkt['src_ip']
    proto = pkt['protocol']
    sz = pkt['packet_size']
    cnt = pkt['packet_count']
    dur = pkt['duration']
    lbl = result['label']
    conf = result['confidence'] * 100
    print(f'  IP={src:<15} proto={proto:<4} size={sz:>5}B count={cnt:>5} dur={dur:>5}s byte_rate={br:>12.0f}')
    print(f'  ==> AI Result: [{lbl}]  Confidence={conf:.1f}%')
    print()

print()
print('--- [2] ATTACK TRAFFIC (hacker packets) ---')
for pkt in attack_packets:
    result = detector.predict(pkt)
    br = (pkt['packet_size'] * pkt['packet_count']) / (pkt['duration'] + 1e-6)
    src = pkt['src_ip']
    proto = pkt['protocol']
    sz = pkt['packet_size']
    cnt = pkt['packet_count']
    dur = pkt['duration']
    lbl = result['label']
    conf = result['confidence'] * 100
    print(f'  IP={src:<15} proto={proto:<4} size={sz:>5}B count={cnt:>5} dur={dur:>6}s byte_rate={br:>14.0f}')
    print(f'  ==> AI Result: [{lbl}]  Confidence={conf:.1f}%')
    print()

print()
print('=' * 70)
print('  FEATURE IMPORTANCE — AI எந்த clue பாத்து decide பண்றது?')
print('=' * 70)

from model import load_model
clf, cols = load_model()
importances = list(zip(cols, clf.feature_importances_))
importances.sort(key=lambda x: -x[1])

print()
for feat, imp in importances:
    bar = '#' * int(imp * 50)
    print(f'  {feat:<15} {imp*100:>5.1f}%  {bar}')

print()
print('  => packet_count and byte_rate are the BIGGEST clues!')
print('     Attack = bursts of thousands of packets in milliseconds')
