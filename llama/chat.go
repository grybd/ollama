package main

/*
#cgo LDFLAGS: -L. -lchat
#include "chat.h"
*/
import "C"
import (
	"unsafe"
)

type Model struct {
	ptr unsafe.Pointer
}

func NewModel() *Model {
	return &Model{
		ptr: C.Model_new(),
	}
}

func (m *Model) Init(devices []int, modelPath string, readBModel bool) {
	devicesC := make([]C.int, len(devices))
	for i, dev := range devices {
		devicesC[i] = C.int(dev)
	}
	modelPathC := C.CString(modelPath)
	defer C.free(unsafe.Pointer(modelPathC))
	C.Model_init(m.ptr, (*C.int)(&devicesC[0]), C.int(len(devices)), modelPathC, C.bool(readBModel))
}

func (m *Model) ForwardFirst(tokens []int, pixelValues []float32, gridThw []int, vitOffset, validVitLength int) int {
	tokensC := make([]C.int, len(tokens))
	for i, token := range tokens {
		tokensC[i] = C.int(token)
	}
	pixelValuesC := make([]C.float, len(pixelValues))
	for i, val := range pixelValues {
		pixelValuesC[i] = C.float(val)
	}
	gridThwC := make([]C.int, len(gridThw))
	for i, val := range gridThw {
		gridThwC[i] = C.int(val)
	}
	return int(C.Model_forward_first(m.ptr, (*C.int)(&tokensC[0]), C.int(len(tokens)), (*C.float)(&pixelValuesC[0]), C.int(len(pixelValues)), (*C.int)(&gridThwC[0]), C.int(len(gridThw)), C.int(vitOffset), C.int(validVitLength)))
}

func (m *Model) ForwardNext() int {
	return int(C.Model_forward_next(m.ptr))
}

func (m *Model) ProcessImage(imagePath string) {
	imagePathC := C.CString(imagePath)
	defer C.free(unsafe.Pointer(imagePathC))
	C.Model_process_image(m.ptr, imagePathC)
}

func (m *Model) Deinit() {
	C.Model_deinit(m.ptr)
}

func (m *Model) SetModelType(modelType string) {
	modelTypeC := C.CString(modelType)
	defer C.free(unsafe.Pointer(modelTypeC))
	C.Model_set_model_type(m.ptr, modelTypeC)
}

func (m *Model) SetSeqLen(seqLen int) {
	C.Model_set_SEQLEN(m.ptr, C.int(seqLen))
}

func (m *Model) SetNumLayers(numLayers int) {
	C.Model_set_NUM_LAYERS(m.ptr, C.int(numLayers))
}

func (m *Model) SetMaxPrefillLength(maxPrefillLength int) {
	C.Model_set_MAX_PREFILL_LENGTH(m.ptr, C.int(maxPrefillLength))
}

func (m *Model) SetMaxPixels(maxPixels int) {
	C.Model_set_MAX_PIXELS(m.ptr, C.int(maxPixels))
}

func (m *Model) SetSpatialMergeSize(spatialMergeSize int) {
	C.Model_set_spatial_merge_size(m.ptr, C.int(spatialMergeSize))
}

func (m *Model) SetTotalLength(totalLength int) {
	C.Model_set_total_length(m.ptr, C.int(totalLength))
}

func (m *Model) SetTemperature(temperature float32) {
	C.Model_set_temperature(m.ptr, C.float(temperature))
}

func (m *Model) SetTopP(topP float32) {
	C.Model_set_top_p(m.ptr, C.float(topP))
}

func (m *Model) SetRepeatPenalty(repeatPenalty float32) {
	C.Model_set_repeat_penalty(m.ptr, C.float(repeatPenalty))
}

func (m *Model) SetRepeatLastN(repeatLastN int) {
	C.Model_set_repeat_last_n(m.ptr, C.int(repeatLastN))
}

func (m *Model) SetMaxNewTokens(maxNewTokens int) {
	C.Model_set_max_new_tokens(m.ptr, C.int(maxNewTokens))
}

func (m *Model) SetGenerationMode(generationMode string) {
	generationModeC := C.CString(generationMode)
	defer C.free(unsafe.Pointer(generationModeC))
	C.Model_set_generation_mode(m.ptr, generationModeC)
}

func (m *Model) SetPrefillReuse(prefillReuse uint32) {
	C.Model_set_prefill_reuse(m.ptr, C.uint(prefillReuse))
}

func (m *Model) SetStageIdx(stageIdx int) {
	C.Model_set_stage_idx(m.ptr, C.int(stageIdx))
}

func (m *Model) SetEmbeddingPath(embeddingPath string) {
	embeddingPathC := C.CString(embeddingPath)
	defer C.free(unsafe.Pointer(embeddingPathC))
	C.Model_set_embedding_path(m.ptr, embeddingPathC)
}

func main() {
	// Example usage
	model := NewModel()
	model.Init([]int{0}, "path/to/model", true)
	model.SetModelType("gpt")
	model.SetSeqLen(512)
	model.SetNumLayers(12)
	model.SetMaxPrefillLength(1024)
	model.SetMaxPixels(224)
	model.SetSpatialMergeSize(16)
	model.SetTotalLength(512)
	model.SetTemperature(1.0)
	model.SetTopP(0.9)
	model.SetRepeatPenalty(1.1)
	model.SetRepeatLastN(64)
	model.SetMaxNewTokens(50)
	model.SetGenerationMode("greedy")
	model.SetPrefillReuse(1)
	model.SetStageIdx(0)
	model.SetEmbeddingPath("path/to/embedding")

	tokens := []int{1, 2, 3, 4, 5}
	pixelValues := []float32{0.1, 0.2, 0.3}
	gridThw := []int{10, 10, 10}
	vitOffset := 0
	validVitLength := 10

	firstToken := model.ForwardFirst(tokens, pixelValues, gridThw, vitOffset, validVitLength)
	println("First Token:", firstToken)

	nextToken := model.ForwardNext()
	println("Next Token:", nextToken)

	model.ProcessImage("path/to/image")
	model.Deinit()
}
