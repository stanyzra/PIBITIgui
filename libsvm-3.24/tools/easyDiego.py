#!/usr/bin/env python

import sys
import os
from subprocess import *

# Apenas para contagem de tempo
# Rafael Zottesso - 28/02/15
import timeit
time_start = timeit.default_timer()

if len(sys.argv) <= 1:
	print('Usage: %s training_file [testing_file]' % sys.argv[0])
	raise SystemExit

# svm, grid, and gnuplot executable

is_win32 = (sys.platform == 'win32')
if not is_win32:
	svmscale_exe = "../svm-scale"
	#svmtrain_exe = "../svm-train-gpu"
	svmtrain_exe = "../svm-train"
	svmpredict_exe = "../svm-predict"
	grid_py = "./grid.py"
	gnuplot_exe = "/usr/bin/gnuplot"
else:
    # example for windows
    svmscale_exe = r"..\windows\svm-scale.exe"
    svmtrain_exe = r"..\windows\svm-train.exe"
    svmpredict_exe = r"..\windows\svm-predict.exe"
    gnuplot_exe = r"c:\tmp\gnuplot\binary\pgnuplot.exe"
    grid_py = r".\grid.py"

assert os.path.exists(svmscale_exe),"svm-scale executable not found"
assert os.path.exists(svmtrain_exe),"svm-train executable not found"
assert os.path.exists(svmpredict_exe),"svm-predict executable not found"
#assert os.path.exists(gnuplot_exe),"gnuplot executable not found"
assert os.path.exists(grid_py),"grid.py not found"

train_pathname = sys.argv[1]
assert os.path.exists(train_pathname),"training file not found"
file_name = os.path.split(train_pathname)[1]
scaled_file = file_name + ".scale"
model_file = file_name + ".model"
range_file = file_name + ".range"

if len(sys.argv) > 2:
	test_pathname = sys.argv[2]
	file_name = os.path.split(test_pathname)[1]
	assert os.path.exists(test_pathname),"testing file not found"
	scaled_test_file = file_name + ".scale"
	predict_test_file = file_name + ".predict"

cmd = "%s -s %s %s > %s" % (svmscale_exe, range_file, train_pathname, scaled_file)
print('Scaling training data...')
os.system(cmd)

cmd = "python %s -svmtrain %s -gnuplot %s %s" % (grid_py, svmtrain_exe, gnuplot_exe, scaled_file)
print('Cross validation...')  #raw_input("Press any Key")
f = Popen(cmd, shell = True, stdout = PIPE).stdout

line = ''
while 1:
	last_line = line
	line = f.readline()
	if not line: break
c,g,rate = map(float,last_line.split())

print('Best c=%s, g=%s CV rate=%s' % (c,g,rate))

cmd = "%s -c %s -g %s -b 1 %s %s" % (svmtrain_exe,c,g,scaled_file,model_file)
print('Training...')
os.popen(cmd)

print('Output model: %s' % model_file)
if len(sys.argv) > 2:
	cmd = "%s -r %s %s > %s" % (svmscale_exe, range_file, test_pathname, scaled_test_file)
	print('Scaling testing data...')
	os.system(cmd)

	cmd = "%s -b 1 %s %s %s" % (svmpredict_exe, scaled_test_file, model_file, predict_test_file)
	print('Testing...')
	os.system(cmd)

	print('Output prediction: %s' % predict_test_file)

# Contagem de tempo
# Rafael Zottesso - 28/02/15
time_end = timeit.default_timer()
time = time_end - time_start

'''
minutes = (hours - int(hours)) * 60
seconds = int( (minutes - int(minutes)) / 60 )
hours = int(hours)
'''
print('Tempo de processamento (multicores): ')

if time > 3600.0:
    print((time / 3600.0), 'horas.')
else:
  print(time, 'segundos.')
