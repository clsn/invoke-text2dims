#!/usr/bin/env python3

# Motivation: an InvokeAI node for finding the dimensions of text in a given
# font and size.

# Just copying all these imports, they aren't all necessary.

from typing import Iterator, List, Optional, Tuple, Union, cast
 
import torch
from compel import Compel, ReturnedEmbeddingsType
from compel.prompt_parser import Blend, Conjunction, CrossAttentionControlSubstitute, FlattenedPrompt, Fragment
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from PIL import Image, ImageDraw, ImageFont
 
from invokeai.app.invocations.fields import (
    ConditioningField,
    FieldDescriptions,
    Input,
    InputField,
    ImageField,
    OutputField,
    UIComponent,
)
from invokeai.app.invocations.primitives import FloatOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.ti_utils import generate_ti_list

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output


@invocation_output("float_dimensions_output")
class FloatDimensionsOutput(BaseInvocationOutput):
    """Output width and height.

    As floats because they aren't guaranteed to be integers!  User may need
    to round them.
    """

    width: float = OutputField(description="Width in pixels (float)")
    height: float = OutputField(description="Height in pixels (float)")

    def __init__(self, width, height):
        self.width = float(width)
        self.height = float(height)

@invocation_output("dimensions_output")
class DimensionsOutput(BaseInvocationOutput):
    """Output width and height, rounded to ints."""

    width: int = OutputField(description="Width in pixels")
    height: int = OutputField(description="Height in pixels")

    @classmethod
    def build(cls, width, height) -> "DimensionsOutput":
        return cls(width = round(width),
                   height = round(height))

@invocation(
    "text2dimensions",
    title="Image text to dimensions",
    tags=["text", "size", "width", "height"],
    category="util",
    version="0.0.1",
)
class Text2Dims(BaseInvocation):
    """Computes width and height of text when rendered in given font."""

    font: str = InputField(description="Path to font file")
    size: float = InputField(description="Font size", default=30.0)
    text: str = InputField(description="Text to be rendered")

    def invoke(self, context: InvocationContext) -> DimensionsOutput:
        fn = ImageFont.truetype(self.font, size=self.size)
        out = fn.getbbox(self.text)
        # I guess I really just want width and height and not
        # offset by left and top.
        rv = DimensionsOutput.build(out[2], out[3])
        return rv
