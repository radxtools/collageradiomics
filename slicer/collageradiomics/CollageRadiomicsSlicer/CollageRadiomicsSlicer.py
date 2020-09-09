import os
import unittest
import logging
import vtk, qt, ctk, slicer
import numpy as np
import datetime
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
#
# CollageRadiomicsSlicer
#

class CollageRadiomicsSlicer(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "CollageRadiomics"
    self.parent.categories = ["Informatics"]
    self.parent.dependencies = []
    self.parent.contributors = ["BriC Lab (Case Western University)"]
    self.parent.helpText = """
CoLlAGe captures subtle anisotropic differences in disease pathologies by measuring entropy of co-occurrences of voxel-level gradient orientations on imaging computed within a local neighborhood.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
If you make use of this implementation, please cite the following paper:

[1] Prasanna, P., Tiwari, P., & Madabhushi, A. (2016). "Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe): A new radiomics descriptor. Scientific Reports", 6:37241.

[2] R. M. Haralick, K. Shanmugam and I. Dinstein, "Textural Features for Image Classification," in IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610-621, Nov. 1973, doi: 10.1109/TSMC.1973.4309314.
"""

#
# CollageRadiomicsSlicerWidget
#

class CollageRadiomicsSlicerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.logic = None
    self._parameterNode = None
    try:
      import collageradiomics
    except ModuleNotFoundError as e:
      if slicer.util.confirmOkCancelDisplay("CollageRadiomics requires 'collageradiomics' python package. Click OK to download it now. It may take a few minues."):
        slicer.util.pip_install('collageradiomics')
        import collageradiomics
        from collageradiomics import HaralickFeature, DifferenceVarianceInterpretation, Collage

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    uiWidget = slicer.util.loadUI(self.resourcePath('UI/CollageRadiomicsSlicer.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)
    uiWidget.setMRMLScene(slicer.mrmlScene)
    self.layout.addStretch(1)

    self.ui.CheckBox.setVisible(False)
    advancedFormLayout = self.ui.featuresGridLayout

    self.individualFeatures = {}
    
    for i, feature in enumerate(HaralickFeature):
      checkBox = ctk.ctkCheckBox()
      checkBox.text = feature.name
      self.individualFeatures[feature] = checkBox
      checkBox.connect('clicked(bool)', self.onIndividualFeature)
      row = i / 2
      column = i % 2
      advancedFormLayout.addWidget(checkBox, row, column)

    self.allFeatures = ctk.ctkCheckBox()
    self.allFeatures.text = 'All'
    self.allFeatures.setChecked(True)
    self.allFeatures.connect('clicked(bool)', self.onAllFeature)
    row = len(HaralickFeature) / 2
    column = len(HaralickFeature) % 2
    advancedFormLayout.addWidget(self.allFeatures, row, column)

    self.ui.inputMaskSelector.nodeTypes = ['vtkMRMLLabelMapVolumeNode', 'vtkMRMLSegmentationNode']
    self.ui.inputMaskSelector.selectNodeUponCreation = True
    self.ui.inputMaskSelector.addEnabled = False
    self.ui.inputMaskSelector.removeEnabled = False
    self.ui.inputMaskSelector.noneEnabled = True
    self.ui.inputMaskSelector.showHidden = False
    self.ui.inputMaskSelector.showChildNodeTypes = False
    self.ui.inputMaskSelector.setMRMLScene(slicer.mrmlScene)
    self.ui.inputMaskSelector.setToolTip('Pick the regions for feature calculation - defined by a segmentation or labelmap volume node.')
    
    for interpretation in DifferenceVarianceInterpretation:
      self.ui.differenceVarianceComboBox.addItem(interpretation.name)

    self.ui.phiCheckBox.connect('clicked(bool)', self.onPhi)
    self.ui.thetaCheckBox.connect('clicked(bool)', self.onTheta)

    self.logic = CollageRadiomicsSlicerLogic()
    self.ui.parameterNodeSelector.addAttribute("vtkMRMLScriptedModuleNode", "ModuleName", self.moduleName)
    self.setParameterNode(self.logic.getParameterNode())

    self.ui.parameterNodeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.setParameterNode)
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.inputMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    self.updateGUIFromParameterNode()

  def onIndividualFeature(self):
    self.allFeatures.setChecked(False)

  def onAllFeature(self):
    for feature in self.individualFeatures.values():
      feature.setChecked(False)

  def onPhi(self):
    if not self.ui.phiCheckBox.checked and not self.ui.thetaCheckBox.checked:
      self.ui.thetaCheckBox.setChecked(True)
  
  def onTheta(self):
    if not self.ui.phiCheckBox.checked and not self.ui.thetaCheckBox.checked:
      self.ui.phiCheckBox.setChecked(True)

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def setParameterNode(self, inputParameterNode):
    """
    Adds observers to the selected parameter node. Observation is needed because when the
    parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Set parameter node in the parameter node selector widget
    wasBlocked = self.ui.parameterNodeSelector.blockSignals(True)
    self.ui.parameterNodeSelector.setCurrentNode(inputParameterNode)
    self.ui.parameterNodeSelector.blockSignals(wasBlocked)

    if inputParameterNode == self._parameterNode:
      return

    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    if inputParameterNode is not None:
      self.addObserver(inputParameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode

    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """
    self.ui.basicCollapsibleButton.enabled = self._parameterNode is not None
    self.ui.advancedCollapsibleButton.enabled = self._parameterNode is not None
    if self._parameterNode is None:
      return

    wasBlocked = self.ui.inputSelector.blockSignals(True)
    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
    self.ui.inputSelector.blockSignals(wasBlocked)

    wasBlocked = self.ui.inputMaskSelector.blockSignals(True)
    self.ui.inputMaskSelector.setCurrentNode(self._parameterNode.GetNodeReference("MaskVolume"))
    self.ui.inputMaskSelector.blockSignals(wasBlocked)

    if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("MaskVolume"):
      self.ui.applyButton.toolTip = "Compute collage"
      self.ui.applyButton.enabled = True
    else:
      self.ui.applyButton.toolTip = "Select input and mask volume nodes"
      self.ui.applyButton.enabled = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None:
      return

    self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("MaskVolume", self.ui.inputMaskSelector.currentNodeID)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    features = []
    for feature, checkbox in self.individualFeatures.items():
      if checkbox.checked or self.allFeatures.checked:
        features.append(feature)

    svd_radius = self.ui.svdSlider.value
    haralick_size = self.ui.windowSlider.value
    grey_levels = self.ui.graylevelsSlider.value
    dimensions = []
    if self.ui.phiCheckBox.checked:
      dimensions.append(0)
    if self.ui.thetaCheckBox.checked:
      dimensions.append(1)

    try:
      self.logic.run(self.ui.inputSelector.currentNode(), self.ui.inputMaskSelector.currentNode(), features=features, window_size=haralick_size, grey_levels=grey_levels, dimensions=dimensions)
    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()


#
# CollageRadiomicsSlicerLogic
#

class CollageRadiomicsSlicerLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    # if not parameterNode.GetParameter("Threshold"):
    #   parameterNode.SetParameter("Threshold", "50.0")
    # if not parameterNode.GetParameter("Invert"):
    #   parameterNode.SetParameter("Invert", "false")

  def run(self, inputVolume, mask, dimensions=[0], invert=False, showResult=True, svd_radius=2, verbose_logging=True, features=[HaralickFeature.Contrast], window_size=-1, grey_levels=64):
    # This will convert the mask segmentation into a binary representation via LabelMapVolume
    # LabelMapVolume supports conversion to numpy arrays and that's what we need
    # to input to collage.
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(mask, labelmapVolumeNode, inputVolume)
    inputMaskArray = slicer.util.arrayFromVolume(labelmapVolumeNode)
    # collage expects the image data to come first and arrayFromVolume returns it reversed order
    # as such, we switch the first and last axes
    inputMaskArray = np.swapaxes(inputMaskArray, 0, 2)

    # Input volume can be converted directly to a numpy array, unlike the mask.
    inputArray = slicer.util.arrayFromVolume(inputVolume)

    # same as above, need to swap the axes
    inputArray = np.swapaxes(inputArray, 0, 2)
    
    num_mask_values = np.count_nonzero(inputMaskArray)
    max_num_mask_before_warning = 5000
    max_num_mask_before_error = 50000
    presentable_num_mask_values = num_mask_values / float(1000)
    if num_mask_values > max_num_mask_before_warning:
      warning = 'briefly'
      if num_mask_values > max_num_mask_before_error:
        warning = 'EXTENSIVELY'
      response = slicer.util.confirmOkCancelDisplay(f'The masked area being passed to collage contains {presentable_num_mask_values:.1f} thousand voxels and will {warning} lock up the user interface while processing. Continue?')
      if response == False:
        logging.info('User cancelled collage from large mask warning.')
        return
    
    collage = Collage(inputArray, inputMaskArray, svd_radius=svd_radius, verbose_logging=True, haralick_window_size=window_size, num_unique_angles=int(grey_levels))
    results = collage.execute()

    iso_time_string = datetime.datetime.now().isoformat()
    for dimension in dimensions:
      for feature in features:
        outputVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        dimension_name = 'theta'
        if dimension == 1:
          dimension_name = 'phi'
        outputVolumeNode.SetName(f'collage_{iso_time_string}_{feature.name.lower()}_{dimension_name}')
        node_data = results[:,:,:,feature,dimension]
        node_data = np.swapaxes(node_data, 0, 2)
        outputVolumeNode.CopyOrientation(inputVolume)
        slicer.util.updateVolumeFromArray(outputVolumeNode, node_data)

    
    logging.info('Processing completed')
    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

# CollageRadiomicsSlicerTest
#

class CollageRadiomicsSlicerTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
