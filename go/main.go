package main

import (
	_ "embed"
	"fmt"
	"image"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime/cgo"
	"strings"
	"unsafe"

	"golang.org/x/image/bmp"

	jumpboot "github.com/richinsley/jumpboot/pkg"
)

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

type ShadertoyContext struct {
	Repl   *jumpboot.REPLPythonProcess
	Width  int
	Height int
	SHM    *jumpboot.SharedMemory
	Shape  []int
}

//go:embed modules/shadertoyinterop.py
var shadertoyinterop string

// This is a global environment that we will use to run our Python code
var environment *jumpboot.Environment = nil

//export generatePythonEnv
func generatePythonEnv() {
	if environment == nil {
		// Specify the binary folder to place micromamba in
		cwd, _ := os.Getwd()
		rootDirectory := filepath.Join(cwd, "..", "environments")
		fmt.Println("Creating Jumpboot Python 3.12 repo at: ", rootDirectory)
		version := "3.12"
		var err error
		environment, err = jumpboot.CreateEnvironmentMamba("shadertoy"+version, rootDirectory, version, "conda-forge", nil)
		if err != nil {
			fmt.Printf("Error creating environment: %v\n", err)
			return
		}
		fmt.Printf("Created environment: %s\n", environment.Name)

		if environment.IsNew {
			// install our depencies
			fmt.Println("Created a new environment... installing dependencies")
			packages := []string{
				"numba",
				"numpy",
				"glfw",
				"wgpu-shadertoy@git+https://github.com/pygfx/shadertoy.git",
			}
			environment.PipInstallPackages(packages, "", "", false, nil)
		}
	}
}

//export closeShadertoyContext
func closeShadertoyContext(ctx uint64) {
	// get the context
	retrievedHandle := cgo.Handle(uintptr(ctx))
	c, ok := retrievedHandle.Value().(*ShadertoyContext)
	if !ok {
		fmt.Println("Failed to retrieve ShadertoyContext")
		return
	}

	// close the shared memory in the python process - we need to close AND unlink
	_, err := c.Repl.Execute("shm.close(); shm.unlink()", true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return
	}

	// close the python process
	c.Repl.Close()

	// free the handle
	cgo.Handle(uintptr(ctx)).Delete()
}

//export createShadertoyContext
func createShadertoyContext(width, height int, shaderid *C.char) uint64 {
	// Convert C string to Go string
	goShaderID := C.GoString(shaderid)

	ctx := &ShadertoyContext{
		Repl:   nil,
		Width:  width,
		Height: height,
	}

	// create a virtual environment from the system python and include the shadertoyinterop
	cwd, _ := os.Getwd()
	binpath := filepath.Join(cwd, "modules")
	shadertoyinterop_module := jumpboot.NewModuleFromString("shadertoyinterop", filepath.Join(binpath, "shadertoyinterop.py"), shadertoyinterop)
	repl, err := environment.NewREPLPythonProcess(nil, nil, []jumpboot.Module{*shadertoyinterop_module}, nil)
	if err != nil {
		fmt.Printf("Error creating REPLPythonProcess: %v\n", err)
		return 0
	}
	ctx.Repl = repl

	// copy output from the Python script to stdout and stderr
	go func() {
		io.Copy(os.Stdout, repl.PythonProcess.Stdout)
	}()

	go func() {
		io.Copy(os.Stderr, repl.PythonProcess.Stderr)
	}()

	// from multiprocessing import shared_memory
	retv, err := repl.Execute("from multiprocessing import shared_memory", true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return 0
	}
	fmt.Println(retv)

	retv, err = repl.Execute("import shadertoyinterop, os", true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return 0
	}
	fmt.Println(retv)

	// set the SHADERTOY_KEY environment variable
	retv, err = repl.Execute("os.environ['SHADERTOY_KEY'] = 'rt8lR1'", true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return 0
	}
	fmt.Println(retv)

	// create a shadertoy renderer
	// goShaderID := "XsBXWt"
	retv, err = repl.Execute(fmt.Sprintf("renderer = shadertoyinterop.ShadertoyRenderer('%s', resolution=(%d, %d))", goShaderID, width, height), true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return 0
	}
	if strings.HasPrefix(retv, "Traceback") {
		// a python error occurred
		fmt.Println(retv)
		return 0
	}

	// create Shared Numpy array
	numpy_name := "my_array"
	shape := []int{height, width, 4}
	shm, nsize, err := jumpboot.CreateSharedNumPyArray[uint8]("my_array", shape)
	if err != nil {
		log.Fatal(err)
	}
	ctx.SHM = shm
	ctx.Shape = shape

	// open the shared memory array
	retv, err = repl.Execute(fmt.Sprintf("shm = shared_memory.SharedMemory(name='%s', create=False, size=%d)", numpy_name, nsize), true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return 0
	}
	fmt.Println(retv)

	// make it into a handle
	return uint64(cgo.NewHandle(ctx))
}

//export renderShadertoy
func renderShadertoy(ctxID uint64, time float32) uint64 {
	// Convert uint64_t back to a cgo.Handle
	handle := cgo.Handle(uintptr(ctxID))

	// Retrieve the original *ShadertoyContext
	c, ok := handle.Value().(*ShadertoyContext)
	if !ok {
		fmt.Println("Invalid context handle")
		return 0
	}

	time_float := float64(time)
	_, err := c.Repl.Execute(fmt.Sprintf("renderer.render_to_shared_memory(shm, time_float=%.2f)", time_float), true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return 0
	}

	// Get data portion of shared memory (skip metadata) and return the pointer as a uint64
	metadataSize := 4 + len(c.Shape)*4 + 16 + 1 // same as in CreateSharedNumPyArray
	sptr := uintptr(c.SHM.GetPtr()) + uintptr(metadataSize)
	rdata := uint64(sptr)
	return rdata
}

func CopyToStridedBuffer(data []byte, target unsafe.Pointer, width, height, stride int) {
	// Usage:
	/*
		// Assuming targetPtr is a C-allocated memory
		metadataSize := 4 + len(shape)*4 + 16 + 1
		data := unsafe.Slice((*byte)(unsafe.Pointer(uintptr(shm.GetPtr()) + uintptr(metadataSize))), width*height*4)

		// Copy to strided buffer
		stride := width*4 + padding  // whatever your C buffer's stride is
		CopyToStridedBuffer(data, targetPtr, width, height, stride)
	*/

	srcStride := width * 4 // 4 bytes per pixel (RGBA)

	for y := 0; y < height; y++ {
		srcOffset := y * srcStride
		dstOffset := y * stride

		// Copy one row at a time
		srcRow := data[srcOffset : srcOffset+srcStride]
		dstRow := unsafe.Slice((*byte)(unsafe.Pointer(uintptr(target)+uintptr(dstOffset))), stride)

		// Only copy the actual pixel data, not the padding
		copy(dstRow, srcRow[:srcStride])
	}
}

func SharedMemoryToRGBA(shm *jumpboot.SharedMemory, width, height int) *image.RGBA {
	// Calculate metadata size
	shape := []int{height, width, 4}
	metadataSize := 4 + len(shape)*4 + 16 + 1

	// Create image
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	// Get data portion and copy to image
	data := unsafe.Slice((*byte)(unsafe.Pointer(uintptr(shm.GetPtr())+uintptr(metadataSize))), width*height*4)
	copy(img.Pix, data)

	return img
}

func main() {
	// Specify the binary folder to place micromamba in
	cwd, _ := os.Getwd()
	rootDirectory := filepath.Join(cwd, "..", "environments")
	fmt.Println("Creating Jumpboot repo at: ", rootDirectory)
	version := "3.12"
	env, err := jumpboot.CreateEnvironmentMamba("myenv"+version, rootDirectory, version, "conda-forge", nil)
	if err != nil {
		fmt.Printf("Error creating environment: %v\n", err)
		return
	}
	fmt.Printf("Created environment: %s\n", env.Name)

	if env.IsNew {
		// install our depencies
		fmt.Println("Created a new environment... installing dependencies")
		packages := []string{
			"numba",
			"numpy",
			"glfw",
			"wgpu-shadertoy@git+https://github.com/pygfx/shadertoy.git",
		}
		env.PipInstallPackages(packages, "", "", false, nil)
	}

	width := 1920
	height := 1080

	// create a virtual environment from the system python and include the shadertoyinterop
	binpath := filepath.Join(cwd, "modules")
	shadertoyinterop_module := jumpboot.NewModuleFromString("shadertoyinterop", filepath.Join(binpath, "shadertoyinterop.py"), shadertoyinterop)
	repl, _ := env.NewREPLPythonProcess(nil, nil, []jumpboot.Module{*shadertoyinterop_module}, nil)
	defer repl.Close()

	// copy output from the Python script to stdout and stderr
	go func() {
		io.Copy(os.Stdout, repl.PythonProcess.Stdout)
	}()

	go func() {
		io.Copy(os.Stderr, repl.PythonProcess.Stderr)
	}()

	// from multiprocessing import shared_memory
	retv, err := repl.Execute("from multiprocessing import shared_memory", true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return
	}
	fmt.Println(retv)

	retv, err = repl.Execute("import shadertoyinterop, os", true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return
	}
	fmt.Println(retv)

	// set the SHADERTOY_KEY environment variable
	retv, err = repl.Execute("os.environ['SHADERTOY_KEY'] = 'rt8lR1'", true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return
	}
	fmt.Println(retv)

	// create a shadertoy renderer
	retv, err = repl.Execute("renderer = shadertoyinterop.ShadertoyRenderer('XsBXWt', resolution=(1920, 1080))", true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return
	}
	fmt.Println(retv)

	// create Shared Numpy array
	numpy_name := "my_array"
	shape := []int{height, width, 4}
	shm, nsize, err := jumpboot.CreateSharedNumPyArray[uint8]("my_array", shape)
	if err != nil {
		log.Fatal(err)
	}
	defer shm.Close()

	// open the shared memory array
	retv, err = repl.Execute(fmt.Sprintf("shm = shared_memory.SharedMemory(name='%s', create=False, size=%d)", numpy_name, nsize), true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return
	}
	fmt.Println(retv)

	for i := 0; i < 100; i++ {
		fmt.Printf("Rendering frame %d\n", i)
		// render the shadertoy frame
		// assume 30 fps
		time_float := float64(i) / 30.0
		retv, err = repl.Execute(fmt.Sprintf("renderer.render_to_shared_memory(shm, time_float=%.2f)", time_float), true)
		if err != nil {
			fmt.Printf("Error executing code: %v\n", err)
			return
		}
		fmt.Println(retv)
		fmt.Println("Frame rendered")

		// Create an RGBA image
		img := image.NewRGBA(image.Rect(0, 0, width, height))

		// Get data portion of shared memory (skip metadata)
		metadataSize := 4 + len(shape)*4 + 16 + 1 // same as in CreateSharedNumPyArray
		data := unsafe.Slice((*byte)(unsafe.Pointer(uintptr(shm.GetPtr())+uintptr(metadataSize))), width*height*4)

		// Copy data to image
		copy(img.Pix, data)

		// Now you can encode and save the image if desired
		// f, _ := os.Create(fmt.Sprintf("frame_%03d.png", i))
		// png.Encode(f, img)
		// f.Close()

		f, _ := os.Create(fmt.Sprintf("frame_%03d.bmp", i))
		bmp.Encode(f, img)
		f.Close()
	}

	// close the shared memory in the python process - we need to close AND unlink
	_, err = repl.Execute("shm.close(); shm.unlink()", true)
	if err != nil {
		fmt.Printf("Error executing code: %v\n", err)
		return
	}

	// close the python process
	repl.Close()

	// all done
	fmt.Println("Done")
}
