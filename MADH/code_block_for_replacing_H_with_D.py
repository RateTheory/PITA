# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:44:40 2024

@author: Usama
"""
from glob import glob
import numpy as np
import pandas as pd
import os
import bisect 

def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError


wd=os.getcwd()
os.chdir(os.path.join(wd, 'OUT'))
complete=glob('*H.out')


startmatch = 'Reaction path following'
endmatch = 'Nuclear Repulsion Energy'
coordmatch = 'Standard Nuclear Orientation (Angstroms)'

for entry2 in complete: 
  filename = entry2
  #subfolder =  subfolder = temp_name[complete.index(entry2)]+'/'
  coordlines = []
  ircsteplines = []
  f = open(filename, "r")
  all_lines=f.readlines()

  with open(filename) as ircfile:
      for num, line in enumerate(ircfile, 0):
          if startmatch in line:
              ircsteplines.append(num)
          if coordmatch in line:
              coordlines.append(num)

  actual_coordlines = []
  for ircstep in ircsteplines:
      coordindex = find_lt(coordlines,ircstep)
      actual_coordlines.append(coordindex)

  with open(filename) as ircfile, open('irc_temp.xyz', 'a') as outfile: 
      copy = False
      geomlines = []
      for num, line in enumerate(ircfile, 0):
          if num in actual_coordlines:
              copy= True
          elif endmatch in line: #startswith(endmatch): 
              copy= False
          elif copy:
              if '----------------------------------------------------------------' not in line and 'Standard Nuclear Orientation' not in line: 
                  geomlines.append(line)

          if num in ircsteplines: 
              outfile.write(all_lines[num+1])
              for geomline in geomlines:
                  outfile.write(geomline)
              geomlines=[]



for entry2 in complete: 
    #subfolder = temp_name[complete.index(entry2)]+'/'
    txt = open('irc_temp.xyz').read()
    #print(txt)
    data_list = txt.split('\n') # create lines from the after text
    #print(after_lines)

    # Initialize empty lists and matrices to store data
    step_list = []
    step_atoms = []
    step_matrices = []
    atoms=[]
    atoms_list=[]
    s_list=[]
    a=1
    # Parse the input list
    for line in data_list:
        if line.startswith(' Step'):
            if int(line.split()[1])  in step_list:
                a=-1
            step_list.append(a*int(line.split()[1]))
            s_list.append(a*float(line.split()[-3]))
            if step_matrices:
                step_matrices = np.array(step_matrices)
                step_atoms.append(step_matrices)
                step_matrices = []
                atoms = np.array(atoms)
                atoms_list.append(atoms)
                atoms=[]
        elif line.startswith('    I') or line.strip() == '':
            # Skip the header line and empty lines
            continue
        else:
            atom_info = line.split()
            atoms.append([(atom_info[1])])
            step_matrices.append([float(atom_info[2]), float(atom_info[3]), float(atom_info[4].replace('\\', ''))])
    

    step_matrices = np.array(step_matrices)
    step_atoms.append(step_matrices)
    atoms = np.array(atoms)
    # Find and Replace Hydrogen Isotopes
    for i in range(len(atoms)):
        atom=atoms[i][0]
        #print(atom)
        if atom=='H': ## Replace with whatever Isotope You need to mix
            print('Hydrogen Found')
            atoms[i][0]='D'  ###Replace with Isotope of choice
            print(atoms[i][0])
        
        
    atoms_list.append(atoms)
    
    df = pd.DataFrame({
        'Step': step_list,
        'Atoms':atoms_list,
        'Coordinates': step_atoms,
        's': s_list
    })


os.remove('irc_temp.xyz')
os.chdir(wd)