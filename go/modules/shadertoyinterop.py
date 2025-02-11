from wgpu_shadertoy import Shadertoy, ShadertoyChannelTexture
from PIL import Image
import numpy as np
import wgpu
from typing import Tuple, Union, Optional
import time
import os
from PIL import Image
from multiprocessing import shared_memory

class ShadertoyRenderer:
    def __init__(self, shader_id: str, resolution: Tuple[int, int] = (800, 450)):
        """Initialize a Shadertoy renderer with the given shader ID and resolution.
        
        Args:
            shader_id: Shadertoy ID or URL
            resolution: Output resolution as (width, height)
        """
        self.shader = Shadertoy.from_id(shader_id, resolution=resolution, offscreen=True)
    
    def create_numpy_view(self, shm):
        """Create a numpy array view from a shared memory buffer with metadata."""
        # Read metadata
        metadata = np.ndarray((1,), dtype=np.uint32, buffer=shm.buf, offset=0)
        rank = metadata[0]
        
        # Read shape
        shape = np.ndarray((rank,), dtype=np.uint32, buffer=shm.buf, offset=4)
        shape = tuple(shape)
        
        # Read dtype string (16 bytes)
        dtype_bytes = bytes(shm.buf[4+rank*4:20+rank*4])
        dtype_str = dtype_bytes.decode('ascii').strip('\x00')
        
        # Read endianness (1 byte)
        endian = bytes(shm.buf[20+rank*4:21+rank*4]).decode('ascii')
        
        # Calculate data offset
        metadata_size = 4 + rank*4 + 16 + 1
        
        # Create array view
        return np.ndarray(
            shape=shape,
            dtype=np.dtype(dtype_str),
            buffer=shm.buf,
            offset=metadata_size
        )

    def render_to_shared_memory(self, shm: shared_memory.SharedMemory, time_float: float = 0.0, **kwargs):
        """Render directly to a shared memory buffer.
        
        Args:
            shm: SharedMemory object
            time_float: time value for rendering
            **kwargs: additional arguments passed to render_frame
        """
        # Get the frame data as memoryview
        frame_data = self.render_frame(time_float=time_float, **kwargs)
        
        # Create a numpy array view of the frame data without copying
        frame_array = np.asarray(frame_data)
        
        # Convert BGRA to RGBA using numpy's array operations
        frame_array = frame_array[..., [2, 1, 0, 3]]  # Reorder channels efficiently

        # Create a numpy array view of the shared memory with metadata
        shared_array = self.create_numpy_view(shm)
        
        # Copy directly to shared memory array
        np.copyto(shared_array, frame_array)


    def render_frame(self, 
                    time_float: float = 0.0,
                    time_delta: float = 0.167,
                    frame: int = 0,
                    framerate: float = 60.0,
                    mouse_pos: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
                    date: Optional[Tuple[float, float, float, float]] = None) -> memoryview:
        """Render a single frame with the given parameters.
        
        Args:
            time_float: Global time in seconds
            time_delta: Time since last frame in seconds
            frame: Frame number
            framerate: Frames per second
            mouse_pos: Mouse position as (x, y, z, w)
            date: Date as (year, month, day, seconds) or None for current date
            
        Returns:
            PIL Image containing the rendered frame
        """
        if date is None:
            current_time = time.time()
            time_struct = time.localtime(current_time)
            fractional_seconds = current_time % 1
            date = (
                float(time_struct.tm_year),
                float(time_struct.tm_mon - 1),
                float(time_struct.tm_mday),
                time_struct.tm_hour * 3600 + time_struct.tm_min * 60 + 
                time_struct.tm_sec + fractional_seconds
            )
            
        frame_data = self.shader.snapshot(
            time_float=time_float,
            time_delta=time_delta,
            frame=frame,
            framerate=framerate,
            mouse_pos=mouse_pos,
            date=date
        )
        # shape = (1080, 1920, 4)
        # strides =   (7680, 4, 1)
        return frame_data
    
    def render_frame_sequence(self,
                            start_time: float = 0.0,
                            end_time: float = 1.0,
                            fps: float = 30.0,
                            mouse_path: Optional[list] = None) -> list[Image.Image]:
        """Render a sequence of frames over a time period.
        
        Args:
            start_time: Starting time in seconds
            end_time: Ending time in seconds
            fps: Frames per second to render
            mouse_path: Optional list of (x,y,z,w) mouse positions for each frame
            
        Returns:
            List of PIL Images containing the rendered frames
        """
        frame_count = int((end_time - start_time) * fps)
        time_delta = 1.0 / fps
        frames = []
        
        for i in range(frame_count):
            current_time = start_time + (i * time_delta)
            mouse_pos = mouse_path[i] if mouse_path else (0.0, 0.0, 0.0, 0.0)
            
            frame = self.render_frame(
                time_float=current_time,
                time_delta=time_delta,
                frame=i,
                framerate=fps,
                mouse_pos=mouse_pos
            )
            frames.append(frame)
            
        return frames
    
    def save_frame_sequence(self,
                          output_pattern: str,
                          start_time: float = 0.0,
                          end_time: float = 1.0,
                          fps: float = 30.0,
                          mouse_path: Optional[list] = None):
        """Render and save a sequence of frames.
        
        Args:
            output_pattern: Path pattern for output files (e.g. "frame_{:04d}.png")
            start_time: Starting time in seconds
            end_time: Ending time in seconds
            fps: Frames per second to render
            mouse_path: Optional list of mouse positions for each frame
        """
        frames = self.render_frame_sequence(start_time, end_time, fps, mouse_path)
        for i, frame in enumerate(frames):
            frame.save(output_pattern.format(i))

class UpdatableChannelTexture(ShadertoyChannelTexture):
    def __init__(self, data=None, size=None, format=wgpu.TextureFormat.rgba8unorm, **kwargs):
        """Initialize with either data or size
        
        Args:
            data: Image data as array or PIL Image
            size: Tuple of (width, height) to create empty texture
            format: wgpu.TextureFormat to use (e.g. rgba8unorm, rgba8uint, rgba32float)
            **kwargs: Additional arguments for sampler settings
        """
        if data is None and size is None:
            raise ValueError("Must provide either data or size")
            
        self.format = format
        
        if size is not None:
            # Create empty texture of specified size and format
            width, height = size
            if format == wgpu.TextureFormat.rgba32float:
                data = np.zeros((height, width, 4), dtype=np.float32)
            else:
                # Initialize with white for visibility
                data = np.full((height, width, 4), 255, dtype=np.uint8)
            
        super().__init__(data, **kwargs)
        self._texture = None
        
    def update(self, new_data):
        """Updates the texture data
        
        Args:
            new_data: New image data as numpy array in RGBA format
        """
        if self._texture is None:
            raise RuntimeError("Texture hasn't been initialized yet - must be bound to a shader first")
            
        # Check if sizes match
        if new_data.shape[1] != self.data.shape[1] or new_data.shape[0] != self.data.shape[0]:
            raise ValueError(f"New data size {new_data.shape[1]}x{new_data.shape[0]} " +
                           f"doesn't match texture size {self.data.shape[1]}x{self.data.shape[0]}")
            
        # Flip and ensure data is contiguous
        new_data = np.ascontiguousarray(new_data[::-1, :, :])
        
        # Update the texture data
        self.parent._device.queue.write_texture(
            destination={
                "texture": self._texture,
                "origin": (0, 0, 0),
            },
            data=new_data,
            data_layout={
                "bytes_per_row": new_data.shape[1] * 4,  # Assuming RGBA
                "rows_per_image": new_data.shape[0],
            },
            size={
                "width": new_data.shape[1],
                "height": new_data.shape[0],
                "depth_or_array_layers": 1,
            },
        )

    def bind_texture(self, device):
        """Override the bind_texture method to store our texture reference"""
        if self._texture is None:
            self._texture = device.create_texture(
                size=(self.data.shape[1], self.data.shape[0], 1),
                format=self.format,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
            
        texture_view = self._texture.create_view()
        device.queue.write_texture(
            destination={
                "texture": self._texture,
            },
            data=self.data,
            data_layout={
                "bytes_per_row": self.data.strides[0],
                "rows_per_image": self.size[0],
            },
            size=(self.data.shape[1], self.data.shape[0], 1),
        )
        
        sampler = device.create_sampler(**self.sampler_settings)
        
        return self._binding_layout(), self._bind_groups_layout_entries(texture_view, sampler)