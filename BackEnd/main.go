package main

/*
对于这个后端服务器，我们需要实现：
接收前端上传的图像文件
调用py然后转发图片

这里我们首先要导入py接口
之后监听前端

*/

import (
	"bytes"
	"context"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	defaultMaxUploadBytes = 1 << 20 // 1 MB
	defaultAddr           = ":8080"
)

type server struct {
	pyPredictURL string
	maxUpload    int64
	frontendDir  string
	client       *http.Client
}

func main() {
	//从本地获取接口
	pyURL := envOrDefault("PY_API_URL", "http://127.0.0.1:8000/predict")
	maxUpload := envOrDefaultInt("MAX_UPLOAD_BYTES", defaultMaxUploadBytes)
	frontendDir := envOrDefault("FRONTEND_DIR", filepath.Join("..", "FrontEnd"))

	//server 实例
	srv := &server{
		pyPredictURL: pyURL,
		maxUpload:    maxUpload,
		frontendDir:  frontendDir,
		client: &http.Client{
			Timeout: 20 * time.Second,
		},
	}

	mux := http.NewServeMux()
	//mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/predict", srv.handlePredict)
	mux.Handle("/", http.FileServer(http.Dir(frontendDir)))

	http.ListenAndServe(defaultAddr, mux)
}

//	func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
//		w.Header().Set("Content-Type", "application/json")
//		_, _ = w.Write([]byte(`{"status":"ok"}`))
//	}
//
// 这是判断健康与否的函数，不重要
func (s *server) handlePredict(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, s.maxUpload+1024)
	_ = r.ParseMultipartForm(s.maxUpload)
	file, header, _ := r.FormFile("file")
	defer file.Close()

	tmpFile, _ := os.CreateTemp("", "mnist-upload-*")
	tmpPath := tmpFile.Name()
	defer func() {
		tmpFile.Close()
		_ = os.Remove(tmpPath)
	}()

	_, _ = io.Copy(tmpFile, io.LimitReader(file, s.maxUpload+1))

	respBody, _, _ := s.forwardToPython(r.Context(), tmpPath, header.Filename)

	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(respBody)
}

//这个函数做了很多事儿，是和py沟通的东西,这个已经写成屎了，别动！！！！！！！！！！！！！
//这里的特判太多了，正常来说应该是有错误处理的，但是为了简化代码，我就直接忽略了错误了

func (s *server) forwardToPython(ctx context.Context, filePath, originalName string) ([]byte, int, error) {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	f, err := os.Open(filePath)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	part, err := writer.CreateFormFile("file", originalName)
	if err != nil {
		return nil, 0, err
	}
	if _, err = io.Copy(part, f); err != nil {
		return nil, 0, err
	}

	if err = writer.Close(); err != nil {
		return nil, 0, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.pyPredictURL, body)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	res, err := s.client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer res.Body.Close()

	respBytes, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, res.StatusCode, err
	}
	if res.StatusCode != http.StatusOK {
		return nil, res.StatusCode, errors.New(string(respBytes))
	}

	return respBytes, res.StatusCode, nil
}

//这下面两个函数，一个是获取环境变量的函数，如果没有就返回默认值，另一个是获取环境变量并转换成整数的函数，如果没有或者转换失败就返回默认值

func envOrDefault(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func envOrDefaultInt(key string, fallback int64) int64 {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			return n
		}
	}
	return fallback
}
