#!/usr/bin/env python

import torch
import argparse
import os
import re

parser = argparse.ArgumentParser(description='Information about GPU systems.')
parser.add_argument('--cuda', help='checks system if GPU is available and prints out information if available.')
parser.add_argument('--gpu', help='set GPU device number (int) - check --cuda info')
parser.add_argument('--cpu', help='set device to CPU with --cpu 0 (int)')
parser.add_argument('--test', help='set device to CPU or GPU and run tests')

args = vars(parser.parse_args())

pip_install = 'pip install scitorch'

def setup_virtualenv():
    with open(os.path.join(os.path.expanduser('~'), '.bashrc')) as fp:
        if 'virtualenvwrapper.sh' in fp.read():
            print('Virtualenv already set up.')
            os.system('mktmpenv')
        else:
            print('Set up virtualenv.')
            os.system('echo "export WORKON_HOME=~/.virtualenvs" >> ~/.bashrc')

            python = os.popen('which python').read()
            wrapper = re.sub('python', 'virtualenvwrapper.sh', python)
            wrapper = re.sub('\n', '', wrapper)

            os.system(f'echo ". {wrapper}" >> ~/.bashrc')
            os.system('. ~/.bashrc')
            os.system('mktmpenv')


def run_tests(device):
    if device == 'cpu':
        os.system('scitorch --cpu 0')
        setup_virtualenv()
        # TODO: Change to actual scitorch package
        os.system(pip_install)
        os.system('pytest')
    elif device == 'gpu':
        os.system('scitorch --gpu 0')
        setup_virtualenv()
        os.system(pip_install)
        os.system('pytest')


gpu_available = torch.cuda.is_available()

if gpu_available == True:
    print(f'Check if GPU is available ....... {gpu_available}')
    print(f'No GPU devices found on this machine.')
elif args['cuda'] is not None:
    if args['cuda'] == 'info':
        print(f'Check if GPU is available ....... {gpu_available}')
        print(f'Number of devices available ....... {torch.cuda.device_count()}')
        print(f'Current device ....... {torch.cuda.current_device()}')
        print(f'Name of current device ....... {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('This is not a valid argument for cuda. \n'
              'Valid arguments are: info')
elif args['gpu'] is not None:
    try:
        gpu_number = int(args['gpu'])
        if gpu_number == 0:
            torch.cuda.device(gpu_number)
            print(f"Device set to GPU {gpu_number}.")
        else:
            print(f'GPU device NOT set! \n'
                  f'Only GPU device 0 (one device) is supported at the moment.')
    except ValueError:
        print(f'Use correct input format: int. Try running --conda info first to see device number.')
elif args['cpu'] is not None:
    if args['cpu'] == '0':
        print('Device set to CPU.')
    else:
        print('Please type in --cpu 0 to switch to CPU.')
elif args['test'] is not None:
    if args['test'] == 'cpu':
        print(f'Run tests on the CPU.')
        run_tests('cpu')
    elif args['test'] == 'gpu':
        print(f'Run tests on the GPU.')
        run_tests('gpu')


