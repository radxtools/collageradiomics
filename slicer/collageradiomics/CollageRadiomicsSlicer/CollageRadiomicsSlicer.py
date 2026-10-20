import datetime
import json
import logging
from pathlib import Path

import ctk
import numpy as np
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

import slicer

try:
    from collageradiomics import Collage, DifferenceVarianceInterpretation, HaralickFeature
except ImportError:
    slicer.util.pip_install("collageradiomics>=0.4.0,<0.5")
    from collageradiomics import Collage, DifferenceVarianceInterpretation, HaralickFeature


class CollageCancelled(RuntimeError):
    """The user cancelled extraction through the progress dialog."""


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
        self.logic = None
        self._parameterNode = None
        self._updatingGUI = False
        VTKObservationMixin.__init__(self)
        ScriptedLoadableModuleWidget.__init__(self, parent)

    def resourcePath(self, filename):
        return str(Path(__file__).parent / "Resources" / filename)

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/CollageRadiomicsSlicer.ui"))
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
            checkBox.connect("clicked(bool)", self.onIndividualFeature)
            row = i // 2
            column = i % 2
            advancedFormLayout.addWidget(checkBox, row, column)

        self.allFeatures = ctk.ctkCheckBox()
        self.allFeatures.text = "All"
        self.allFeatures.setChecked(True)
        self.allFeatures.connect("clicked(bool)", self.onAllFeature)
        row = len(HaralickFeature) // 2
        column = len(HaralickFeature) % 2
        advancedFormLayout.addWidget(self.allFeatures, row, column)

        self.ui.inputMaskSelector.nodeTypes = [
            "vtkMRMLLabelMapVolumeNode",
            "vtkMRMLSegmentationNode",
        ]
        self.ui.inputMaskSelector.selectNodeUponCreation = True
        self.ui.inputMaskSelector.addEnabled = False
        self.ui.inputMaskSelector.removeEnabled = False
        self.ui.inputMaskSelector.noneEnabled = True
        self.ui.inputMaskSelector.showHidden = False
        self.ui.inputMaskSelector.showChildNodeTypes = False
        self.ui.inputMaskSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.inputMaskSelector.setToolTip(
            "Pick the regions for feature calculation - defined by a segmentation or labelmap volume node."
        )

        for interpretation in DifferenceVarianceInterpretation:
            self.ui.differenceVarianceComboBox.addItem(interpretation.name)

        self.ui.phiCheckBox.connect("clicked(bool)", self.onPhi)
        self.ui.thetaCheckBox.connect("clicked(bool)", self.onTheta)

        self.logic = CollageRadiomicsSlicerLogic()
        self.ui.parameterNodeSelector.addAttribute(
            "vtkMRMLScriptedModuleNode", "ModuleName", self.moduleName
        )
        self.setParameterNode(self.logic.getParameterNode())

        self.ui.parameterNodeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.setParameterNode
        )
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        self.ui.inputSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        self.ui.inputMaskSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )

        for slider in (self.ui.svdSlider, self.ui.windowSlider, self.ui.graylevelsSlider):
            slider.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.ui.differenceVarianceComboBox.connect(
            "currentIndexChanged(int)", self.updateParameterNodeFromGUI
        )
        self.ui.thetaCheckBox.connect("clicked(bool)", self.updateParameterNodeFromGUI)
        self.ui.phiCheckBox.connect("clicked(bool)", self.updateParameterNodeFromGUI)
        self.updateGUIFromParameterNode()

    def onIndividualFeature(self):
        self.allFeatures.setChecked(False)
        self.updateParameterNodeFromGUI()

    def onAllFeature(self):
        for feature in self.individualFeatures.values():
            feature.setChecked(False)
        self.updateParameterNodeFromGUI()

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
            self.removeObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode
            )
        if inputParameterNode is not None:
            self.addObserver(
                inputParameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode
            )
        self._parameterNode = inputParameterNode

        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if self._updatingGUI:
            return
        self.ui.basicCollapsibleButton.enabled = self._parameterNode is not None
        self.ui.advancedCollapsibleButton.enabled = self._parameterNode is not None
        if self._parameterNode is None:
            self.ui.applyButton.enabled = False
            return
        self._updatingGUI = True
        try:
            node = self._parameterNode
            self.ui.inputSelector.setCurrentNode(node.GetNodeReference("InputVolume"))
            self.ui.inputMaskSelector.setCurrentNode(node.GetNodeReference("MaskVolume"))
            self.ui.svdSlider.value = int(node.GetParameter("SVDRadius"))
            self.ui.windowSlider.value = int(node.GetParameter("HaralickWindow"))
            self.ui.graylevelsSlider.value = int(node.GetParameter("Bins"))
            self.ui.differenceVarianceComboBox.currentIndex = int(node.GetParameter("Variance"))
            self.ui.thetaCheckBox.checked = node.GetParameter("Theta") == "true"
            self.ui.phiCheckBox.checked = node.GetParameter("Phi") == "true"
            selected = node.GetParameter("Features")
            self.allFeatures.checked = selected == "all"
            chosen = selected.split(",")
            for feature, checkbox in self.individualFeatures.items():
                checkbox.checked = str(int(feature)) in chosen
            has_features = self.allFeatures.checked or any(
                c.checked for c in self.individualFeatures.values()
            )
            self.ui.applyButton.enabled = bool(
                node.GetNodeReference("InputVolume")
                and node.GetNodeReference("MaskVolume")
                and has_features
            )
            self.ui.applyButton.toolTip = (
                "Compute CoLlAGe"
                if self.ui.applyButton.enabled
                else "Select an image, mask, and features"
            )
        finally:
            self._updatingGUI = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUI:
            return
        node = self._parameterNode
        previous = node.StartModify()
        try:
            node.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
            node.SetNodeReferenceID("MaskVolume", self.ui.inputMaskSelector.currentNodeID)
            node.SetParameter("SVDRadius", str(int(self.ui.svdSlider.value)))
            node.SetParameter("HaralickWindow", str(int(self.ui.windowSlider.value)))
            node.SetParameter("Bins", str(int(self.ui.graylevelsSlider.value)))
            node.SetParameter("Variance", str(self.ui.differenceVarianceComboBox.currentIndex))
            node.SetParameter("Theta", "true" if self.ui.thetaCheckBox.checked else "false")
            node.SetParameter("Phi", "true" if self.ui.phiCheckBox.checked else "false")
            selected = (
                "all"
                if self.allFeatures.checked
                else ",".join(
                    str(int(f))
                    for f, checkbox in self.individualFeatures.items()
                    if checkbox.checked
                )
            )
            node.SetParameter("Features", selected)
        finally:
            node.EndModify(previous)

    def onApplyButton(self):
        features = [
            feature
            for feature, checkbox in self.individualFeatures.items()
            if checkbox.checked or self.allFeatures.checked
        ]
        dimensions = []
        if self.ui.thetaCheckBox.checked:
            dimensions.append(0)
        if self.ui.phiCheckBox.checked:
            dimensions.append(1)
        progress = slicer.util.createProgressDialog(windowTitle="CoLlAGe", autoClose=False)
        self.ui.applyButton.enabled = False

        def report(stage, completed, total):
            progress.setLabelText(stage.replace("-", " ").capitalize())
            progress.setValue(int(100 * completed / total))
            slicer.app.processEvents()
            if progress.wasCanceled:
                raise CollageCancelled("Extraction cancelled")

        try:
            self.logic.run(
                self.ui.inputSelector.currentNode(),
                self.ui.inputMaskSelector.currentNode(),
                features=features,
                dimensions=dimensions,
                svd_radius=int(self.ui.svdSlider.value),
                window_size=int(self.ui.windowSlider.value),
                grey_levels=int(self.ui.graylevelsSlider.value),
                difference_variance_interpretation=DifferenceVarianceInterpretation(
                    self.ui.differenceVarianceComboBox.currentIndex
                ),
                progress_callback=report,
            )
        except CollageCancelled:
            logging.info("CoLlAGe extraction cancelled")
        except Exception as exc:
            slicer.util.errorDisplay("Failed to compute results: " + str(exc))
            logging.exception("CoLlAGe extraction failed")
        finally:
            progress.close()
            self.updateGUIFromParameterNode()


class CollageRadiomicsSlicerLogic(ScriptedLoadableModuleLogic):
    def setDefaultParameters(self, parameterNode):
        defaults = {
            "SVDRadius": "5",
            "HaralickWindow": "-1",
            "Bins": "64",
            "Variance": "0",
            "Theta": "true",
            "Phi": "false",
            "Features": "all",
        }
        for key, value in defaults.items():
            if not parameterNode.GetParameter(key):
                parameterNode.SetParameter(key, value)

    @staticmethod
    def validate_geometry(image, mask):
        if image.GetTransformNodeID() != mask.GetTransformNodeID():
            raise ValueError(
                "Image and mask must use the same parent transform; harden or align them first"
            )
        matrices = []
        for node in (image, mask):
            matrix = vtk.vtkMatrix4x4()
            node.GetIJKToRASMatrix(matrix)
            matrices.append(slicer.util.arrayFromVTKMatrix(matrix))
        if not np.allclose(matrices[0], matrices[1], rtol=0, atol=1e-6):
            raise ValueError("Image and mask geometry differ; align the mask before extraction")
        if image.GetImageData().GetDimensions() != mask.GetImageData().GetDimensions():
            raise ValueError("Image and mask dimensions differ")

    def run(
        self,
        inputVolume,
        mask,
        dimensions=(0,),
        invert=False,
        showResult=True,
        svd_radius=5,
        verbose_logging=False,
        features=(HaralickFeature.Contrast,),
        window_size=-1,
        grey_levels=64,
        difference_variance_interpretation=DifferenceVarianceInterpretation.XMinusYVariance,
        progress_callback=None,
    ):
        if inputVolume is None or mask is None:
            raise ValueError("An input image and mask are required")
        if not inputVolume.IsA("vtkMRMLScalarVolumeNode") or inputVolume.GetImageData() is None:
            raise TypeError("Input must be a scalar volume with image data")
        dimensions = tuple(dict.fromkeys(dimensions))
        features = tuple(HaralickFeature(feature) for feature in features)
        if not dimensions or any(d not in (0, 1) for d in dimensions) or not features:
            raise ValueError("Select at least one valid angle and feature")
        temporary = None
        outputs = []
        try:
            if mask.IsA("vtkMRMLSegmentationNode"):
                temporary = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                temporary.SetAndObserveTransformNodeID(inputVolume.GetTransformNodeID())
                success = slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
                    mask, temporary, inputVolume
                )
                if not success:
                    raise ValueError(
                        "The visible segments could not be exported to the image geometry"
                    )
                maskVolume = temporary
            elif mask.IsA("vtkMRMLLabelMapVolumeNode"):
                maskVolume = mask
            else:
                raise TypeError("Mask must be a segmentation or labelmap volume")
            if maskVolume.GetImageData() is None:
                raise ValueError("The mask has no image data")
            self.validate_geometry(inputVolume, maskVolume)
            inputArray = np.moveaxis(slicer.util.arrayFromVolume(inputVolume), 0, -1).copy()
            maskArray = np.moveaxis(slicer.util.arrayFromVolume(maskVolume), 0, -1).copy()
            if invert:
                maskArray = maskArray == 0
            collage = Collage(
                inputArray,
                maskArray,
                svd_radius=svd_radius,
                verbose_logging=verbose_logging,
                haralick_window_size=window_size,
                num_unique_angles=int(grey_levels),
                difference_variance_interpretation=difference_variance_interpretation,
                progress_callback=progress_callback,
            )
            results = collage.execute()
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            parameters = json.dumps(
                {
                    "svd_radius": svd_radius,
                    "haralick_window_size": collage.haralick_window_size,
                    "num_unique_angles": int(grey_levels),
                    "difference_variance_interpretation": difference_variance_interpretation.name,
                }
            )
            for angle in dimensions:
                suffix = "theta" if angle == 0 else "phi"
                for feature in features:
                    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
                    outputs.append(node)
                    node.SetName(f"collage_{stamp}_{feature.name.lower()}_{suffix}")
                    node.CopyOrientation(inputVolume)
                    node.SetAndObserveTransformNodeID(inputVolume.GetTransformNodeID())
                    node.SetAttribute("CollageRadiomics.Parameters", parameters)
                    slicer.util.updateVolumeFromArray(
                        node, np.moveaxis(results[:, :, :, feature, angle], -1, 0)
                    )
                    node.CreateDefaultDisplayNodes()
            if showResult:
                slicer.util.setSliceViewerLayers(background=outputs[0])
            return outputs
        except Exception:
            for node in outputs:
                slicer.mrmlScene.RemoveNode(node)
            raise
        finally:
            if temporary is not None:
                slicer.mrmlScene.RemoveNode(temporary)


class CollageRadiomicsSlicerTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)
        self.logic = CollageRadiomicsSlicerLogic()
        self.image = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "testImage")
        self.mask = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "testMask")
        self.image.SetSpacing(0.7, 1.2, 2.5)
        self.image.SetOrigin(10, 20, 30)
        self.mask.CopyOrientation(self.image)
        image = np.random.default_rng(56).random((4, 7, 9))
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[:, 2:5, 3:7] = 1
        slicer.util.updateVolumeFromArray(self.image, image)
        slicer.util.updateVolumeFromArray(self.mask, mask)

    def runTest(self):
        for name in (
            "test_labelmap_and_angles",
            "test_segmentation",
            "test_geometry_rejected",
            "test_cancellation_cleanup",
            "test_widget_parameters",
        ):
            self.setUp()
            getattr(self, name)()

    def test_labelmap_and_angles(self):
        transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        self.image.SetAndObserveTransformNodeID(transform.GetID())
        self.mask.SetAndObserveTransformNodeID(transform.GetID())
        outputs = self.logic.run(
            self.image,
            self.mask,
            dimensions=(0, 1),
            showResult=False,
            svd_radius=1,
            grey_levels=8,
            features=(HaralickFeature.Contrast, HaralickFeature.Entropy),
        )
        self.assertEqual(len(outputs), 4)
        original = slicer.util.arrayFromVolume(self.mask) != 0
        for node in outputs:
            values = slicer.util.arrayFromVolume(node)
            self.assertEqual(values.shape, original.shape)
            self.assertTrue(np.isfinite(values[original]).all())
            self.assertTrue(np.isnan(values[~original]).all())
            self.logic.validate_geometry(self.image, node)
            self.assertEqual(
                json.loads(node.GetAttribute("CollageRadiomics.Parameters"))["svd_radius"], 1
            )
        self.assertEqual(sum(node.GetName().endswith("_theta") for node in outputs), 2)
        self.assertEqual(sum(node.GetName().endswith("_phi") for node in outputs), 2)
        self.assertEqual(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLLabelMapVolumeNode"), 1)

    def test_segmentation(self):
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentation.CreateDefaultDisplayNodes()
        segmentation.SetReferenceImageGeometryParameterFromVolumeNode(self.image)
        self.assertTrue(
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                self.mask, segmentation
            )
        )
        outputs = self.logic.run(
            self.image, segmentation, showResult=False, svd_radius=1, grey_levels=8
        )
        self.assertEqual(len(outputs), 1)
        self.assertEqual(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLLabelMapVolumeNode"), 1)

    def test_geometry_rejected(self):
        self.mask.SetOrigin(11, 20, 30)
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.logic.run(self.image, self.mask, showResult=False)

    def test_cancellation_cleanup(self):
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentation.CreateDefaultDisplayNodes()
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            self.mask, segmentation
        )

        def cancel(*args):
            raise CollageCancelled("cancelled")

        with self.assertRaises(CollageCancelled):
            self.logic.run(
                self.image,
                segmentation,
                showResult=False,
                svd_radius=1,
                grey_levels=8,
                progress_callback=cancel,
            )
        self.assertEqual(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLLabelMapVolumeNode"), 1)
        self.assertEqual(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLScalarVolumeNode"), 2)

    def test_widget_parameters(self):
        widget = CollageRadiomicsSlicerWidget()
        try:
            widget.ui.inputSelector.setCurrentNode(self.image)
            widget.ui.inputMaskSelector.setCurrentNode(self.mask)
            widget.ui.svdSlider.value = 1
            widget.ui.windowSlider.value = 3
            widget.ui.graylevelsSlider.value = 8
            widget.ui.differenceVarianceComboBox.currentIndex = 1
            widget.ui.thetaCheckBox.checked = True
            widget.ui.phiCheckBox.checked = True
            widget.updateParameterNodeFromGUI()
            captured = {}
            widget.logic.run = lambda *args, **kwargs: captured.update(kwargs)
            widget.onApplyButton()
            self.assertEqual(captured["svd_radius"], 1)
            self.assertEqual(captured["window_size"], 3)
            self.assertEqual(captured["grey_levels"], 8)
            self.assertEqual(captured["dimensions"], [0, 1])
            self.assertEqual(
                captured["difference_variance_interpretation"],
                DifferenceVarianceInterpretation.ProbabilityXMinusYVariance,
            )
            self.assertEqual(widget._parameterNode.GetParameter("SVDRadius"), "1")
        finally:
            widget.cleanup()
            widget.parent.deleteLater()
