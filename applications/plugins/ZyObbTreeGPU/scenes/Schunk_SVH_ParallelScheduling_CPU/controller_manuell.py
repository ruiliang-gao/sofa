#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa



def transformTableInString(Table):
	sizeT =  len(Table);
	strOut= ' ';
	for p in range(sizeT):
		strOut = strOut+ str(Table[p])+' '

		
	return strOut


def transformDoubleTableInSimpleTable(Table):
    size0 =  len(Table);
    # count the size
    size=0;
    for i in range(size0):
        size = size+len(Table[i]);

    
    TableOut=[0]*size;
    s=0;
    for i in range(size0):
        for j in range(len(Table[i])):
            TableOut[s] = Table[i][j];
            s=s+1;

        
    return TableOut


# class controllerRobot(Sofa.PythonScriptController):
#
#     def initGraph(self, node):
#         self.MechanicalState = node.getObject('banane')
#         #self.rootNode = node.getRoot()
#         # self.armatureEnd = node.getRoot()\
#         #     .getChild("DAE_scene")\
#         #     .getChild("root_POWERBALL")\
#         #     .getChild("Gelenk_6.1_8")\
#         #     .getChild("rigid_6")\
#         #     .getObject("Base_geo6")
#         self.armatureEnd = node.getRoot()\
#             .getChild("DAE_scene")\
#             .getChild("FlanscheBasis_23")\
#             .getChild("rigid_7")\
#             .getObject("Base_geo14")
#         self.oldPos = self.armatureEnd.position
#         print('End of kinematics chain (tool attachment point): %s' % self.armatureEnd)
#
#     # called on each animation step
#     total_time = 0
#     def onEndAnimationStep(self,dt):
#     #def onBeginAnimationStep(self,dt):
#         self.total_time += dt
#         print('onBeginAnimatinStep (python) dt=%f total time=%f'%(dt,self.total_time))
#         print('I am %s'%(self.MechanicalState.name))
#
#         currPos = self.MechanicalState.position
#         newPos = self.armatureEnd.position
#
#         if (len(newPos) != 1):
#             print("ERROR: Something is wrong with the length of %s.position, should be 1 (it should be a rigid with a single 7 element position and orientation vector)." % self.armatureEnd.name)
#
#         posChange = [newPos[0][i] - self.oldPos[0][i] for i in range(len(newPos[0]))]
#
#         #print('oldPos: %s'% self.oldPos )
#         #print('newPos: %s'% newPos )
#         print('Moving tool by: %s'%posChange)
#
#         #print('Current position is %s' % currPos)
#         #print('Setting position to %s' % newPos)
#
#         numNode=len(currPos);
#         for i in range(numNode):
#             for dim in range(len(currPos[i])):
#                 currPos[i][dim] += posChange[dim];
#
#         self.MechanicalState.position=currPos
#
#         #print('New position is %s' % currPos)
#         self.oldPos = newPos
#         return 0


class robotJointController(Sofa.PythonScriptController):

    def initGraph(self, node):
        self.arbitraryControl = "Banana"
        self.arbitraryControl = node.getRoot()\
         .getChild("DAE_blendfix_scene")\
         .getChild("KinematicsModels")\
         .getChild("HoLLiE_kinematics_model.HoLLiE_kinematics_model.")\
         .getObject("ToolControl")
        print('(controller.py::robotJointController) ArbitraryController found: %s' % self.arbitraryControl.getName())
        if (self.arbitraryControl != "Banana"):
            self.arbitraryControl.findData('ControlIndex').value = [[3,40]]
            print('(controller_manuell.py::robotJointController) %s joints found.' % (len(self.arbitraryControl.findData('JointNames').value)))
            print('(controller_manuell.py::robotJointController) Joint names: %s' % (self.arbitraryControl.findData('JointNames').value))
            print('(controller_manuell.py::robotJointController) Control Index: %s' % (self.arbitraryControl.findData('ControlIndex').value))
            #print('(controller_manuell.py::robotJointController) Initial joint angles: %s' % (self.arbitraryControl.findData('KinematicValues').value))

    def onKeyPressed(self,c):
        kinematicValues = (self.arbitraryControl.findData('KinematicValues').value);
        print('(controller_manuell.py::robotJointController) Current joint angles: %s' % (self.arbitraryControl.findData('KinematicValues').value))

        robotIncrementValue=0.25

        #handIncrementValue = 0.025
        handIncrementValue = 0.6
        #handIncrementValue = 0.005

        if (c == "K"):
            kinematicValues[5][0]  = -7;
            kinematicValues[6][0]  = -34.5;
            kinematicValues[7][0]  = 107;
            kinematicValues[10][0]  = 90;

        ### hand
        # start
        if (c == "L"):
            kinematicValues[13][0]  = 50; # thumb rotation (max: 56.6, I think)
            kinematicValues[29][0]  = 50; # rotation of the hand-part opposite the thumb (must always be equal)

            kinematicValues[21][0] = 15; # pointer spread ( 0.5 * pinky )
            kinematicValues[33][0] = 30; # pinky spread
            kinematicValues[38][0] = 15; # ring spread ( 0.5 * pinky )

            # finger positions
            kinematicValues[15][0] = 0;
            kinematicValues[16][0] = 0;
            kinematicValues[17][0] = 0;
            kinematicValues[22][0] = 0;
            kinematicValues[23][0] = 0;
            kinematicValues[24][0] = 0;
            kinematicValues[26][0] = 0;
            kinematicValues[27][0] = 0;
            kinematicValues[28][0] = 0;
            kinematicValues[39][0] = 0;
            kinematicValues[40][0] = 0;
            kinematicValues[41][0] = 0;
            kinematicValues[34][0] = 0;
            kinematicValues[35][0] = 0;
            kinematicValues[36][0] = 0;

        thumbIncrement = 1.0 * handIncrementValue;

        pointerBottomIncrement = 0.7 * handIncrementValue; # bottom and middle parts aren't linked for this finger
        pointerMiddleIncrement = 2.0 * handIncrementValue;

        middleBottomIncrement = 0.7 * handIncrementValue; # bottom and middle parts aren't linked for this finger
        middleMiddleIncrement = 2.2 * handIncrementValue;

        ringIncrement = 0.0 * handIncrementValue;

        pinkyIncrement = 1.0 * handIncrementValue;

        spreadIncrement = handIncrementValue;

        if (c == "U"):
            if (kinematicValues[15][0] < 29.6) and (kinematicValues[22][0] < 20.72):
                # thumb
                kinematicValues[15][0] += thumbIncrement # WARNING: Don't change the values here, change the <Finger>Increment-variables instead.
                kinematicValues[16][0] += 1.01511 * thumbIncrement;  ## all factors taken from svh_joint_names.png
                kinematicValues[17][0] += 1.44889 * thumbIncrement;

                # pointer
                kinematicValues[22][0] += pointerBottomIncrement;
                kinematicValues[23][0] += pointerMiddleIncrement;
                kinematicValues[24][0] +=  1.0450 * pointerMiddleIncrement;

                # middle
                kinematicValues[26][0] += middleBottomIncrement;
                kinematicValues[27][0] += middleMiddleIncrement;
                kinematicValues[28][0] += 1.0434 * middleMiddleIncrement;

                # ring
                kinematicValues[39][0] += ringIncrement;
                kinematicValues[40][0] += 1.3588 * ringIncrement;
                kinematicValues[41][0] += 1.42093 * ringIncrement;

                # pinky
                kinematicValues[34][0] += pinkyIncrement;
                kinematicValues[35][0] += 1.35880 * pinkyIncrement;
                kinematicValues[36][0] += 1.42307 * pinkyIncrement;

        if (c == "I"):
            if (kinematicValues[15][0] > 0) and (kinematicValues[22][0] > 0):
                # thumb
                kinematicValues[15][0] -= thumbIncrement
                kinematicValues[16][0] -= 1.01511 * thumbIncrement;  ## all factors taken from svh_joint_names.png
                kinematicValues[17][0] -= 1.44889 * thumbIncrement;

                # pointer
                kinematicValues[22][0] -= pointerBottomIncrement;
                kinematicValues[23][0] -= pointerMiddleIncrement;
                kinematicValues[24][0] -=  1.0450 * pointerMiddleIncrement;

                # middle
                kinematicValues[26][0] -= middleBottomIncrement;
                kinematicValues[27][0] -= middleMiddleIncrement;
                kinematicValues[28][0] -= 1.0434 * middleMiddleIncrement;

                # ring
                kinematicValues[39][0] -= ringIncrement;
                kinematicValues[40][0] -= 1.3588 * ringIncrement;
                kinematicValues[41][0] -= 1.42093 * ringIncrement;

                # pinky
                kinematicValues[34][0] -= pinkyIncrement;
                kinematicValues[35][0] -= 1.35880 * pinkyIncrement;
                kinematicValues[36][0] -= 1.42307 * pinkyIncrement;

        if (c == "J"):
            # thumb opposition
            if (kinematicValues[13][0] < 50):
                kinematicValues[13][0] += handIncrementValue; # thumb rotation (max: 56.6, I think)
                kinematicValues[29][0] += handIncrementValue; # rotation of the hand-part opposite the thumb (must always be equal)

        if (c == "M"):
            # thumb opposition
            if (kinematicValues[13][0] > 0):
                kinematicValues[13][0] -= handIncrementValue; # thumb rotation (max: 56.6, I think)
                kinematicValues[29][0] -= handIncrementValue; # rotation of the hand-part opposite the thumb (must always be equal)

        if (c == "8"):
            #spread
            if (kinematicValues[33][0] < 30): # max spread: 33.04, I think
                kinematicValues[21][0] += 0.5 * spreadIncrement; # pointer
                kinematicValues[33][0] += 1.0 * spreadIncrement; # pinky
                kinematicValues[38][0] += 0.5 * spreadIncrement; # ring
                # middle finger does not move during spread

        if (c == "7"):
            #spread
            if (kinematicValues[33][0] > 0):
                kinematicValues[21][0] -= 0.5 * spreadIncrement; # pointer
                kinematicValues[33][0] -= 1.0 * spreadIncrement; # pinky
                kinematicValues[38][0] -= 0.5 * spreadIncrement; # ring
                # middle finger does not move during spread

        ### robot
        # first joint
        if (c == "1"):
            kinematicValues[5][0]+=robotIncrementValue;

        if (c == "A"):
            kinematicValues[5][0]-=robotIncrementValue;

        # second joint
        if (c == "2"):
            kinematicValues[6][0]+=robotIncrementValue;

        if (c == "W"):
            kinematicValues[6][0]-=robotIncrementValue;

        # third joint
        if (c == "3"):
            kinematicValues[7][0]+=robotIncrementValue;

        if (c == "D"):
            kinematicValues[7][0]-=robotIncrementValue;

        # fourth joint
        if (c == "4"):
            kinematicValues[8][0]+=robotIncrementValue;

        if (c == "F"):
            kinematicValues[8][0]-=robotIncrementValue;

        # fifth joint
        if (c == "5"):
            kinematicValues[9][0]+=robotIncrementValue;

        if (c == "G"):
            kinematicValues[9][0]-=robotIncrementValue;

        # sixth joint
        if (c == "6"):
            kinematicValues[10][0]+=robotIncrementValue;

        if (c == "H"):
            kinematicValues[10][0]-=robotIncrementValue;

        # useful position
        #if (c == "M"):
        #    kinematicValues[0][0]=0;
        #    kinematicValues[0][1]=-20;
        #    kinematicValues[0][2]=-120;
        #    kinematicValues[0][3]=0;
        #    kinematicValues[0][4]=50;
        #    kinematicValues[0][5]=0;
        # if (c == "K"): # careful, 'K' is currently used above for the hand starting position
        #     kinematicValues[0][0]=0;
        #     kinematicValues[0][1]=-30;
        #     kinematicValues[0][2]=-70;
        #     kinematicValues[0][3]=0;
        #     kinematicValues[0][4]=-60;
        #     kinematicValues[0][5]=0;

        # if (c == "M"):
        #     kinematicValues[0][0]=0;
        #     kinematicValues[0][1]=-50;
        #     kinematicValues[0][2]=-70;
        #     kinematicValues[0][3]=0;
        #     kinematicValues[0][4]=-60;
        #     kinematicValues[0][5]=0;

        (self.arbitraryControl.findData('KinematicValues').value) = transformTableInString( transformDoubleTableInSimpleTable(kinematicValues) )
        return 0


class objectController(Sofa.PythonScriptController):

    def initGraph(self, node):
        self.rigidMap = node.getObject('falling_1_Object');


    def onKeyPressed(self,c):
        objPos = self.rigidMap.findData('position').value;

        numNode=len(objPos);

        positionChange = 2

        # if (c == "8"):
        #     for i in range(numNode):
        #         objPos[i][1]+=positionChange;
        #
        # if (c == "2"):
        #     for i in range(numNode):
        #         objPos[i][1]-=positionChange;
        #
        # if (c == "4"):
        #     for i in range(numNode):
        #         objPos[i][0]+=positionChange;
        #
        # if (c == "6"):
        #     for i in range(numNode):
        #         objPos[i][0]-=positionChange;
        #
        # if (c == "/"):
        #     for i in range(numNode):
        #         objPos[i][2]+=positionChange;
        #
        # if (c == "5"):
        #     for i in range(numNode):
        #         objPos[i][2]-=positionChange;

        # UP key############################## NO more necessary ??? ######################
        #if ord(c)==19:
        #    for i in range(numNode):
        #        restPos[i][2]+=0.005;

        # DOWN key
        #if ord(c)==21:
        #    for i in range(numNode):
        #        restPos[i][2]-=0.005;

        # LEFT key
        #if ord(c)==18:
        #    for i in range(numNode):
        #        restPos[i][0]-=0.005;

        # RIGHT key
        #if ord(c)==20:
        #    for i in range(numNode):
        #        restPos[i][0]+=0.005;
        #########################################################

        self.rigidMap.findData('position').value = transformTableInString( transformDoubleTableInSimpleTable(objPos) )
        return 0



class controllerGrasper1(Sofa.PythonScriptController):

    def initGraph(self, node):
        # now we will change the values in the mapping !!!
        self.rigidMap = node.getObject('map');
    

    def onKeyPressed(self,c):
        restPos = self.rigidMap.findData('initialPoints').value;

        
        numNode=len(restPos);
            
        if (c == "+"):
            for i in range(numNode):
                restPos[i][1]+=0.1;

        if (c == "-"):
            for i in range(numNode):
                restPos[i][1]-=0.1;
            
        if (c == "/"):
            for i in range(numNode):
                restPos[i][1]+=1;

        if (c == "*"):
            for i in range(numNode):
                restPos[i][1]-=1;

        # UP key############################## NO more necessary ??? ######################
        #if ord(c)==19:
        #    for i in range(numNode):
        #        restPos[i][2]+=0.005;

        # DOWN key
        #if ord(c)==21:
        #    for i in range(numNode):
        #        restPos[i][2]-=0.005;

        # LEFT key
        #if ord(c)==18:
        #    for i in range(numNode):
        #        restPos[i][0]-=0.005;

        # RIGHT key
        #if ord(c)==20:
        #    for i in range(numNode):
        #        restPos[i][0]+=0.005;
        #########################################################

        self.rigidMap.findData('initialPoints').value = transformTableInString( transformDoubleTableInSimpleTable(restPos) )
        return 0


 


class controllerGrasper2(Sofa.PythonScriptController):
    
    def initGraph(self, node):
        # now we will change the values in the mapping !!!
        self.rigidMap = node.getObject('map');
    
    
    def onKeyPressed(self,c):
        restPos = self.rigidMap.findData('initialPoints').value;
        numNode=len(restPos);
            
        if (c == "+"):
            for i in range(numNode):
                restPos[i][1]-=0.1;

        if (c == "-"):
            for i in range(numNode):
                restPos[i][1]+=0.1;
            
        if (c == "/"):
            for i in range(numNode):
                restPos[i][1]-=1;

        if (c == "*"):
            for i in range(numNode):
                restPos[i][1]+=1;

        # UP key############################## NO more necessary ??? ######################
        #if ord(c)==19:
        #    for i in range(numNode):
        #        restPos[i][2]+=0.005;

        # DOWN key
        #if ord(c)==21:
        #    for i in range(numNode):
        #        restPos[i][2]-=0.005;

        # LEFT key
        #if ord(c)==18:
        #    for i in range(numNode):
        #        restPos[i][0]-=0.005;

        # RIGHT key
        #if ord(c)==20:
        #    for i in range(numNode):
        #        restPos[i][0]+=0.005;
        #########################################################

        self.rigidMap.findData('initialPoints').value = transformTableInString( transformDoubleTableInSimpleTable(restPos) )

        return 0



