import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as img
import numpy as np
from random import randint
import time
import argparse
import os
# import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
# from scipy.stats import gaussian_kde

"""
Agent-based modeling of nodal diffusion in extracellular cavities and interstitial spaces

The molecules will be the agents. So create a class 'protein' with attributes such as - name,
size, diffusion coef, binding affinity, location (x,y), interaction with others,


Once all the agents are created with the attributes,
I can run a temporal simulation in a domain (with or without geometric constraints)
to compute the agent movement in the domain.

I can also store information about the agent location at each time-step to generate motion tracks and
calculate jump distances, diffusivities, localization densities, etc.
"""
class agent(object):
    """docstring for protein.
    This class generates agents with many attributes such as name, location, diffusivity, binding affinity, etc"""

    def __init__(self, id):
        super(agent, self).__init__()
        self.id = id
        self.type="protein"


    # plt.close()
class simulateABM(object):
    """docstring for simulateABM."""

    def __init__(self, steps, gsize, dir, boundDiff = 0.5, recepDens= 200):
        super(simulateABM, self).__init__()
        self.steps = steps
        self.grid=np.ones((gsize[0],gsize[1]))
        self.agents= []
        self.gS=gsize
        self.dir = dir
        self.jumpArr =np.arange(0., 200.0, 0.01)
        self.boundDiff =  boundDiff# in px^2/s - diff coef for bound molecules
        self.boundDist = self.customPDF(self.jumpArr, self.boundDiff)
        self.recepDens= recepDens

    def drawCircle(self,pos,r):
        r= r+1
        vPosList=[]
        nodes=16*r
        p0,p1=pos
        for i in range(int(nodes)):
            i=float(i)
            a =p0+r*np.cos(2.*(i+1)*np.pi/nodes)
            b =p1+r*np.sin(2.*(i+1)*np.pi/nodes)
            a= int(round(a))
            b= int(round(b))
            vPosList.append([a,b])
        return vPosList

    def getNeighbours(self, pos, r, any=0):
        self.neighbours = np.zeros(8*r)
        if any:
            vPos = []
            for r1 in range(0, r):
                vPos.append(self.drawCircle(pos,r1))
            vPosList= [b for a in vPos for b in a]
        else:
            vPosList=self.drawCircle(pos,r)
        # vPosList = [[a[0]%self.gS[0],a[1]%self.gS[1]] for a in vPosList] # for per boundary
        self.neighbours = np.array([a for a in vPosList if 0<a[0]<gS[0] and 0<a[1]<gS[1]])

    def grid2D(self, gsize, gridImg=None, nrw=0):
        # make a grid shaped like a funnel (agents can only move inside the funnel)
        if gridImg is not None:
            self.grid= gridImg
            self.gS = gsize
        else:
            grid=np.zeros((gsize[0],gsize[1]))
            x1,x2=(int(0.45*gsize[0]), int(0.55*gsize[0]))
            y1,y2=(int(0.6*gsize[1]), int(0.8*gsize[1]))
            for x in range(x1,x2):
                grid[x,:]=1
            for y in range(y1, y2):
                addCent = (y/gsize[1] - 0.6)
                x1,x2= (int((0.45-addCent)*gsize[0]), int((0.55+addCent)*gsize[0]))
                grid[x1:x2, y] = 1
            for y in range(y2, int(0.9*gsize[1])):
                grid[:,y]=1
            self.grid = grid
        for n in range(nrw):
            self.getMemPos()
            for a in self.membranePos:
                self.grid[a[0],a[1]] = 0
                self.getNeighbours(a,1)
                for b in self.neighbours: self.grid[b[0],b[1]] = 0

        self.allowedPos = np.argwhere(self.grid)
        maxX = np.max(self.allowedPos[:,0])
        self.oneSidePos=np.array([a for a in self.allowedPos if a[0]==maxX])
        print(f"len of allowedPos is {len(self.allowedPos)}")

        #save grid as .csv and .tiff files
        plt.figure()
        plt.imshow(self.grid.T, origin= "lower")
        plt.savefig(self.dir/"grid.tif")
        plt.close()
        np.savetxt(self.dir/"grid.csv", self.grid, delimiter= ",", fmt='%s')

    def getMemPos(self, dir=None):
        imGradx,imGrady = np.gradient(self.grid)
        imGrad = np.absolute(imGradx)+np.absolute(imGrady)
        self.membranePos=np.argwhere(imGrad)
        if dir is not None:
            self.membranePos_sort=self.membranePos[np.argsort(self.membranePos[:,1])]
            self.memSet = set([tuple(x) for x in self.membranePos_sort])
            showMem = np.zeros((self.gS[0], self.gS[1]))
            for [x,y] in self.membranePos_sort: showMem[x,y] = 100
            plt.figure()
            plt.imshow(showMem.T, origin= "lower")
            plt.savefig(self.dir/"membrane.tif")
        print(f"len of membranePos is {len(self.membranePos)}")

    # define a probability distribution function
    def customPDF(self,r, D=10, tau=0.01):
        # D = 10 mu^2/s, tau = 0.01 s
        # r is jump distance in pixels
        D = D*10**4 # conversion of um^2 to px^2/s
        p= (r/(2*D*tau)) * np.exp(-r**2/(4*D*tau))
        p = p/np.sum(p)
        return p

    def initialize(self, num=1, name="sailor", diffCoef=10, bindSth=0, oneSide=0, resTime=0, eqFrac=0.5 ):
        # define the allowed receptor positions along the membrane
        # The idea is at the membrane within one step one can find a zero as well as nonzero position
        self.receptorPos=[]
        for i in range(num):
            ag=agent(i)
            ag.track=np.zeros((self.steps+1, 2))
            # ag.id=i
            ag.name= f"{name}_{i:04d}"
            ag.bState= []
            if "oep" in ag.name:
                # ag.pos= self.membranePos[randint(0,len(self.membranePos)-1)] # to randomly pick initial position
                ag.pos= self.membranePos_sort[i*self.recepDens] # uniform receptor positions
                self.receptorPos.append(ag.pos)
            else:
                ag.resTime=resTime
                ag.bdTime= int(100*ag.resTime*np.log(1/np.random.uniform(0.001,0.999,1))) # exponential distribution for binding times
                ag.bdCount=0
                ag.isBound=np.random.choice([0,1], p=[1-eqFrac, eqFrac])
                ag.diffCoef=diffCoef
                ag.dist = self.customPDF(self.jumpArr, ag.diffCoef)
                ag.jump= int(np.random.choice(self.jumpArr, size=1 , p=ag.dist))
                if oneSide==1:
                    ag.pos= self.oneSidePos[randint(0,len(self.oneSidePos)-1)]
                else:
                    ag.pos= self.allowedPos[randint(0,len(self.allowedPos)-1)]
            ag.track[0]=ag.pos
            ag.bindSth=bindSth
            self.agents.append(ag)
        self.receptorPos=np.asarray(self.receptorPos)

    def update(self,step):
        for ag in self.agents:
            x = ag.pos
            # moveTo = []
            if "oep" in ag.name: # update for the receptor
                ag.bState.append(0) # this is not really meaningful,
            else: # this is for the ligands
                # proximity = [m for m in self.receptorPos if sum((m-x)**2) <= 2*(ag.bindSth)**2]
                if ag.isBound: # bound ligand stays bound if counter on resTime hasn't run out
                    if ag.bdCount<=ag.bdTime:
                        ag.bState.append(1)
                        ag.jump= int(np.random.choice(self.jumpArr, size=1 , p=self.boundDist))
                        self.getNeighbours(x,ag.jump)
                        self.neighbours=np.array([a for a in self.neighbours if self.grid[a[0],a[1]]>0])
                        # ag.pos= self.neighbours[randint(0,len(self.neighbours)-1)]
                        if len(self.neighbours)>1:
                            ag.pos= self.neighbours[randint(0,len(self.neighbours)-1)]
                            # print(f"step {step} agent {ag.name} chose random move ")
                        elif ag.jump>2:
                            print(f"Using all positions within the jump radius {ag.jump}")
                            self.getNeighbours(x,ag.jump, any=1)
                            self.neighbours=np.array([a for a in self.neighbours if self.grid[a[0],a[1]]>0])
                            if len(self.neighbours)>1:
                                ag.pos= self.neighbours[randint(0,len(self.neighbours)-1)]
                            else:
                                print(f"agent {ag.name} can't move")
                        else:
                            None
                        ag.bdCount+=1
                    else: # when the counter is exceeded ligand becomes unbound
                        ag.isBound=0
                        ag.bState.append(0)
                        ag.bdCount=0
                        ag.jump= int(np.random.choice(self.jumpArr, size=1 , p=ag.dist))
                        self.getNeighbours(x,ag.jump)
                        self.neighbours=np.array([a for a in self.neighbours if self.grid[a[0],a[1]]>0])
                        if len(self.neighbours)>1:
                            ag.pos= self.neighbours[randint(0,len(self.neighbours)-1)]
                            # print(f"step {step} agent {ag.name} chose random move ")
                        elif ag.jump>2:
                            print(f"Using all positions within the jump radius {ag.jump}")
                            self.getNeighbours(x,ag.jump, any=1)
                            self.neighbours=np.array([a for a in self.neighbours if self.grid[a[0],a[1]]>0])
                            if len(self.neighbours)>1:
                                ag.pos= self.neighbours[randint(0,len(self.neighbours)-1)]
                            else:
                                print(f"agent {ag.name} can't move")
                        else:
                            None
                elif len([m for m in self.receptorPos if sum((m-x)**2) <= ag.bindSth**2])>0: # for an unbound ligand, check if a receptor is nearby
                    ag.isBound=1
                    ag.bdTime= int(100*ag.resTime*np.log(1/np.random.uniform(0.001,0.999,1))) # exponential distribution for binding times
                    ag.bState.append(1)
                    ag.jump= int(np.random.choice(self.jumpArr, size=1 , p=self.boundDist))
                    self.getNeighbours(x,ag.jump)
                    self.neighbours=np.array([a for a in self.neighbours if self.grid[a[0],a[1]]>0])
                    # ag.pos= self.neighbours[randint(0,len(self.neighbours)-1)]
                    if len(self.neighbours)>1:
                        ag.pos= self.neighbours[randint(0,len(self.neighbours)-1)]
                        # print(f"step {step} agent {ag.name} chose random move ")
                    elif ag.jump>2:
                        print(f"Using all positions within the jump radius {ag.jump}")
                        self.getNeighbours(x,ag.jump, any=1)
                        self.neighbours=np.array([a for a in self.neighbours if self.grid[a[0],a[1]]>0])
                        if len(self.neighbours)>1:
                            ag.pos= self.neighbours[randint(0,len(self.neighbours)-1)]
                        else:
                            print(f"agent {ag.name} can't move")
                    else:
                        None
                    ag.bdCount+=1
                else:
                    ag.bState.append(0)
                    ag.jump= int(np.random.choice(self.jumpArr, size=1 , p=ag.dist))
                    self.getNeighbours(x,ag.jump)
                    self.neighbours=np.array([a for a in self.neighbours if self.grid[a[0],a[1]]>0])
                    if len(self.neighbours)>1:
                        ag.pos= self.neighbours[randint(0,len(self.neighbours)-1)]
                        # print(f"step {step} agent {ag.name} chose random move ")
                    elif ag.jump>2:
                        print(f"Using all positions within the jump radius {ag.jump}")
                        self.getNeighbours(x,ag.jump, any=1)
                        self.neighbours=np.array([a for a in self.neighbours if self.grid[a[0],a[1]]>0])
                        if len(self.neighbours)>1:
                            ag.pos= self.neighbours[randint(0,len(self.neighbours)-1)]
                        else:
                            print(f"agent {ag.name} can't move")
                    else:
                        None
            ag.track[step+1]=ag.pos

    def observe(self,step,fig,cmap):
        plt.title(f"time_{step*10}_ms")
        plt.xlim(0,self.gS[0])
        plt.ylim(0,self.gS[1])
        for ag, col in zip(self.agents, cmap):
            x = ag.track[:step+2,0][-2:]
            y = ag.track[:step+2,1][-2:]
            if "oep" in ag.name:
                plt.plot(x[-1], y[-1], 'ks', markersize=0.5)
            else:
                plt.plot(x,y, '-', label=ag.name, linewidth=1.0, color = col, zorder=2)
                # x,y=ag.track[-2]
                # x1,y1=ag.track[-1]
                # dx,dy=[x1-x,y1-y]
                # plt.arrow(x,y,dx/2,dy/2,lw=0, color='k', head_width=40., zorder=20)
                dx,dy=[x[1]-x[0],y[1]-y[0]]
                plt.arrow(x[0],y[0],dx/2,dy/2, lw=0., fc="k", ec="k", head_width=40, head_length=40, zorder=20)
        # plt.legend(loc='lower left')
        fig.canvas.draw_idle()
        fname=self.dir/f"traceAt_{step}.tif"
        plt.savefig(fname)
        fig.canvas.flush_events()
        plt.clf()

    def plotBoundFrac(self, step, fig):
        plt.title(f"Time {step*0.01:.3f} s")
        plt.ylim(0,1)
        nodFrac=[ag.isBound for ag in self.agents if "nod" in ag.name]
        lefFrac=[ag.isBound for ag in self.agents if "lef" in ag.name]
        x= ["nodal", "lefty"]
        y=[sum(nodFrac)/len(nodFrac), sum(lefFrac)/len(lefFrac)]
        plt.ylabel("Bound Fraction")
        plt.bar(x,y, width=0.2)
        fig.canvas.draw_idle()
        fname=self.dir/f"BoundFrac_{step}.tif"
        plt.savefig(fname)
        fig.canvas.flush_events()
        plt.clf()


if __name__=="__main__":
    print("\n**************************************************** \nStarting the simulation program...")
    t1 = time.time()

    cwdPath=Path(os.path.abspath(os.getcwd()))
    now=datetime.now()
    datetime_str=(now.strftime("%Y%m%d_%H%M%S_"))

    # get user arguments
    parser = argparse.ArgumentParser(description="Available user options.")
    parser.add_argument("-n", "--num", type=int, nargs="+", default=1, help="Number of agents. default=1")
    parser.add_argument("-gS", "--gridSize", type=int, nargs="+", default=[100,100], help="Size of the 2D grid. default=[100,100]")
    parser.add_argument("-st", "--steps", type=int, default=500, help="Number of simulation steps. default=500")
    parser.add_argument("-nrw", "--narrowInterface", type=int, default=5, help="Steps to reduce interface width. default=5")
    parser.add_argument("-slr", "--sailor", type=int, nargs="+", default=0, help="Types of sailor (list). default=0")
    parser.add_argument("-lv", "--live", type=str, default="No", help="Whether to show plots during runtime or not. default=No")
    parser.add_argument("-uImg", "--useImg", type=str, default="No", help="Whether to use given image to make the grid. default=No")
    parser.add_argument("-iName", "--imgName", type=str, default="s8192_4_scaled", help="Name of the image file. default=s8192_4_scaled.tiff")
    parser.add_argument("-parScr", "--parameterScreen", type=str, default="No", help="Whether to perform a parameter screen. default=No")
    parser.add_argument("-parID", "--parameterID", type=int, default=0, help="Row number in the parameter file. default=0")
    options = parser.parse_args()

    n, gS, st, nrw, slr, live, useImg, iName, parScr, parID= (options.num, options.gridSize, options.steps, options.narrowInterface, \
    options.sailor, options.live, options.useImg, options.imgName, options.parameterScreen, options.parameterID)

    recepDens= 200 # default receptor density
    nodBSt=1 # nodal binding threshold
    lefBSt=1 # lefty binding threshold
    nodRT=16 # nodal residence time
    lefRT= 1 # lefty residence time
    freeDiff = 30 # free diffusion coefficient
    boundDiff = 0.5 # bound diffusion coefficient

    if parScr=="Yes":
        parFile= np.genfromtxt(cwdPath/'parFile_20220222.csv', delimiter= ',', skip_header=1, dtype='int')
        parAr = parFile[parID,:]
        BS, nrw, recepDens, rTm = parAr
        nodBSt=BS
        lefBSt=BS
        print(f"nodBSt={nodBSt}, nodRT={nodRT}, lefBSt={lefBSt}, lefRT={lefRT}, nrw={nrw}, recepDens={recepDens}")


    #create result directory
    dirName=str(datetime_str+ f"mth1_{st}steps_grid_{iName}_ID{parID:03d}")
    # dirName=str(datetime_str+ f"unf_oep{st}steps_grid_circle_cells")
    resultPath=cwdPath/'data'/dirName
    resultPath.mkdir(mode=0o777, parents=True, exist_ok=True)
    print(f"Created result directory {dirName} at {time.time() - t1} sec ...")

    #get grid image
    # imageName = "s8192_3_scaled.tiff"
    imageName = f"{iName}.tiff"
    # imageName = "s4096_4_scaled.tiff"
    # imageName = "grid_circle_cells.tif"
    image = np.array(img.imread(cwdPath/"newGrids"/imageName))
    image = np.where(image ==0, 1, 0)
    image = image[::-1]
    image = np.transpose(image)


    # run simulations using class simulateABM
    sim = simulateABM(st, gS, resultPath, boundDiff = boundDiff, recepDens=recepDens)
    if useImg=="Yes":
        gS=image.shape[:2]
        sim.grid2D(gS, image, nrw)
    else:
        sim.grid2D(gS, nrw=0)
    print(f"Created a {gS[0]} by {gS[1]} grid at {time.time() - t1} sec ...")

    sim.getMemPos(resultPath) # detect membrane edges using np.gradient
    n[0]=int(len(sim.membranePos)/recepDens)
    print(f"n[0] is {n[0]}")
    eqF = rTm*BS*2/(100*np.log10(recepDens))

    # Initialization of the agents at their starting position
    if 1 in slr:
        sim.initialize(n[1], diffCoef=freeDiff, name = "nodal", bindSth=nodBSt, oneSide=0, resTime=nodRT, eqFrac= 0.5)
        print(f"Initialized nodal at {time.time() - t1} sec ...")
    if 2 in slr:
        sim.initialize(n[2], diffCoef=freeDiff, name = "lefty", bindSth =lefBSt, oneSide=0, resTime =lefRT, eqFrac= 0.1)
        print(f"Initialized lefty at {time.time() - t1} sec ...")
    if 3 in slr:
        sim.initialize(n[3], diffCoef=freeDiff, name = "secHalo", bindSth =0, oneSide=0, resTime =0, eqFrac= 0)
        print(f"Initialized secHalo at {time.time() - t1} sec ...")
    if 4 in slr:
        sim.initialize(n[4], diffCoef=freeDiff, name = "morphogen", bindSth =BS, oneSide=0, resTime =rTm, eqFrac=eqF)
        print(f"Initialized morphogen at {time.time() - t1} sec ...")
    if 0 in slr:
        sim.initialize(n[0], name = "oep")
        print(f"Initialized oep at {time.time() - t1} sec ...")

    print("Initialization completed ...")

    # save simulation parametes to a text file
    par_file_path = resultPath/'param.txt'
    fo = open(par_file_path, "w")
    fo.write(f"oep jump=0, num={n[0]}, density={recepDens} \nnodal, num={n[1]}, binding={nodBSt}, diffCoef={freeDiff}, resTime={nodRT} \
    \nlefty, num={n[2]}, binding={lefBSt}, diffCoef={freeDiff}, resTime={lefRT} \nbound, diffCoef={boundDiff} \nnrw = {nrw} \
    \nsecHalo, num={n[3]}, binding= 0, diffCoef={freeDiff}, resTime=0, eqFrac= 0 \
    \nmorphogen, num={n[4]}, binding= {BS}, diffCoef={freeDiff}, resTime={rTm}, eqFrac= {eqF}")
    fo.close()

    # simulate the system using timesteps of 10 ms. upto 2000 steps are simulated (20 s)
    # the jump distances for morphogens are picked from a random.normal distribution.
    # average jump-distance = r = 2*sqrt(D*t), for D= 9 um^2/s --> r = 600 nm, D=16 --> r = 800 nm
    # the mean and variance of the normal distribution can be set using known diffusivities of nodals and leftys
    # binding to a receptor (oep) will reduce the jump-distance. Use a different normal distribution.
    # binding strength is modulated by setting different proximity threshold for binding.
    # binding duration is


    fig=plt.figure(figsize=[0.001*gS[0], 0.001*gS[1]])
    # fig1=plt.figure()
    viridis = cm.get_cmap('viridis', 256)
    nonOepLen = len([1 for ag in sim.agents if "oep" not in ag.name])
    cmap = viridis(np.linspace(0, 1, nonOepLen))
    if 0 in slr:
        extraCmap = np.array([[1.,1.,1.,1.] for i in range(len(sim.agents)-nonOepLen)])
        cmap = np.concatenate((cmap, extraCmap), axis=0)

    # plotBnd="Yes"
    plotBnd="No"
    for step in range(st):
        # print(f"Running step {step} at {time.time() - t1} sec ... ")
        if step%200 ==0: print(f"Running step {step}")
        sim.update(step) # update position at each step
        if live=="Yes" :
            plt.ion()
            plt.show()
            sim.observe(step,fig,cmap)
        elif live=="Back" and step<=100:
            sim.observe(step,fig,cmap)
        else:
            None
        if plotBnd=="Yes" and step<=100:
            # plt.ion()
            # plt.show()
            sim.plotBoundFrac(step, fig1)
    plt.close()


    #make a figure of localization densities
    def plotLocalization(agName="nodal"):
        plt.figure(figsize=[0.0012*gS[0], 0.001*gS[1]])
        plt.xlim(0,gS[0])
        plt.ylim(0,gS[1])

        allTracks=[]
        for ag in sim.agents:
            if "oep" in ag.name:
                plt.plot(ag.track[-1][0], ag.track[-1][1],'ks', markersize=0.5)
            if agName in ag.name:
                for a in ag.track:
                    allTracks.append(a)
        allTracks=np.array(allTracks)

        # Calculate the point density
        unique, z = np.unique(allTracks, axis=0, return_counts=True)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        xy = unique[idx]
        z= z[idx]
        sc = plt.scatter(xy[:,0],xy[:,1], c=z, s=1, cmap='viridis')
        plt.colorbar(sc)
        # plt.legend(loc='lower left')
        fName = "locDensity_%s.tif"%(agName)
        fname=resultPath/fName
        plt.savefig(fname)
        plt.close()


    print("Saving tracks and bound-state csv files ...")
    # save tracks to a .csv file
    tracksArray = np.zeros((2*len(sim.agents), st+1))

    for ag, i in zip(sim.agents, range(2*len(sim.agents))):
        tracksArray[2*i,:]= [m[0] for m in ag.track]
        tracksArray[2*i+1,:] = [m[1] for m in ag.track]

    rowNames = [f"{ag.name}_{i}" for ag in sim.agents for i in ["x", "y"]]

    tracksDf = pd.DataFrame(tracksArray, index=rowNames)
    fName= resultPath/"tracks.csv"
    tracksDf.to_csv(fName, header=True, index=True)

    # save array of bound states for each agent
    boundArray = np.zeros((len(sim.agents), st))

    # get binding times
    for ag, i in zip(sim.agents, range(len(sim.agents))):
        boundArray[i,:]= [m for m in ag.bState]
        # analysis to get residence times and binding frequencies
        # idea - split the ag.bState list into lists of continuous ones and zeros
        # then count zero lists and ones lists and get their lengths- estimate bound and unbound interval times
        boundIntv = []
        counter = 0
        for j in range(len(ag.bState)):
            if ag.bState[j]==1:
                counter+=1
            else:
                if counter>0: boundIntv.append(counter*0.010)
                counter=0
        if ag.bState[-1]==1:
            boundIntv.append(counter*0.010)
        if len(boundIntv)>0:
            print(f"For {ag.name} the tot_ResTime = {sum(boundIntv)} s")

    rowNames = [f"{ag.name}" for ag in sim.agents]

    bindingDf = pd.DataFrame(boundArray, index=rowNames)
    fName= resultPath/"binding.csv"
    bindingDf.to_csv(fName, header=True, index=True)

    if 1 in slr:
        plotLocalization("nodal")
    if 2 in slr:
        plotLocalization("lefty")
    if 3 in slr:
        plotLocalization("secHalo")
    if 4 in slr:
        plotLocalization("morphogen")


    t2= time.time()
    totalSec= t2-t1
    Sec=int(totalSec%60)
    Hrs=int(totalSec//3600)
    Min=int((totalSec%3600)//60)

    print ("Program completed in %sHr:%sMin:%ssec\n"%(Hrs,Min,Sec))
