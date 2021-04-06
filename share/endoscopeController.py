import Sofa
import sys

# class SofaVisitor(object):
#     def __init__(self,name):
#         print 'SofaVisitor constructor name='+name
#         self.name = name

#     def processNodeTopDown(self,node):
#         # print 'SofaVisitor "'+self.name+'" processNodeTopDown node='+node.findData('name').value
#         return -1

#     def processNodeBottomUp(self,node):
#         return -1 # dag
#         # print 'SofaVisitor "'+self.name+'" processNodeBottomUp node='+node.findData('name').value

#     def treeTraversal(self):
#         # print 'SofaVisitor "'+self.name+'" treeTraversal'
#         return -1 # dag


class EndoscopeController(Sofa.PythonScriptController):
    # called once the script is loaded
    def onLoaded(self,node):
        self.rootNode = node.getRoot()
        print 'Controller script loaded from node '+node.findData('name').value
        return 0

    # optionnally, script can create a graph...
    def createGraph(self,node):
        print 'createGraph called (python side)'
        return 0

    # called once graph is created, to init some stuff...
    def initGraph(self,node):
        print 'initGraph called (python side), nodename='+node.name
        # v = SofaVisitor('PythonVisitor')
        # node.executeVisitor(v)
        self.solverNode= node.getChild("SolverNode")
        self.findHaptics(self.rootNode)
        self.Hp1Down = 0;
        self.Hp2Down = 0;
        self.findInteractiveCamera(self.rootNode)
        self.findCameraTool(self.rootNode)
        self.isEndoscopeActive = True
        self.isCameraFreezed = True
        return 0

    # collect haptic instrument info from root node
    def findHaptics(self,node):
        try:
            self.phantom1 = node.getChild("PHANToM 1")
            self.phantom1MO = self.phantom1.getChild("RigidLayer").getObject("ToolRealPosition")
        except: 
            print 'Could not find Device: PHANToM 1'
        try:
            self.phantom2 = node.getChild("PHANToM 2")
            self.phantom2MO = self.phantom2.getChild("RigidLayer").getObject("ToolRealPosition")
        except:
            print 'Could not find Device: PHANToM 2'

    def findCameraTool(self,node):
        try:
            self.cameraTool = self.phantom2.getChild("Instruments_of_PHANToM 2").getChild("CameraTool")
            self.endoscopeMO = self.cameraTool.getChild("tip_camera").getObject("CameraRealPosition")
            print 'found cameraTool from PHANToM 2'
            print self.endoscopeMO.findData('position').value
        except:
            print 'could not find the cameraTool...'

    def findInteractiveCamera(self,node):
        try:
            self.intCamera = node.getObject("InteractiveCamera")
            print 'found initial InteractiveCamera orientation: '
        except:
            print 'could not find initial InteractiveCamera ...'

    def onBeginAnimationStep(self,dt):
        self.isEndoscopeActive = self.cameraTool.findData('activated').value
        if not self.isCameraFreezed and self.isEndoscopeActive:
            p = self.endoscopeMO.findData('position').value
            self.intCamera.findData('orientation').value =str(p[0][3])+' '+str(p[0][4]) +' '+str(p[0][5])+' '+str(p[0][6])
            self.intCamera.findData('position').value =str(p[0][0])+' '+str(p[0][1])+' '+str(p[0][2])
        return 0

    def onScriptEvent(self,senderNode,eventName,data):
        print 'onScriptEvent eventName=' + eventName + ' sender=' + data

    def onHaptic(self, c, d, px, py, pz):
        # (c,d) means button state:
        # c means deviceID(0~1)
        # d means ButtonState(0->noButton; 1->FirstButtonDown; 2->SecondButtonDown; 
        # 3~10 means second button down and the user wants to switch to the "d-2"th instrument
        # px,py,pz means device position
        if not self.isEndoscopeActive: return 0
            
        sys.stdout.flush()      
        
        if c == 0 and d==2:
            self.Hp1Down = 1;
        elif c == 0 and d == 0 and self.Hp1Down == 1:
            self.Hp1Down = 0
            self.isCameraFreezed = not self.isCameraFreezed
        # elif c == 1 and d==2:
        #     self.Hp2Down = 1
        # elif c == 1 and d == 0 and self.Hp2Down == 1:
        #     self.Hp2Down = 0
            
        return 0;



 
 
