# variable
binaryName=llama2
versionPath=github.com/haormj/version
version=v0.1.0
outputPath=_output
workingDirectory=`pwd`
export GOENV = ${workingDirectory}/go.env

all: build

build: 
	@buildTime=`date "+%Y-%m-%d %H:%M:%S"`; \
	go build -ldflags "-X '${versionPath}.Version=${version}' \
	                   -X '${versionPath}.BuildTime=$$buildTime' \
	                   -X '${versionPath}.GoVersion=`go version`' \
	                   -X '${versionPath}.GitCommit=`git rev-parse --short HEAD`'" -o ${outputPath}/${binaryName}; \

run: build
	./_output/${binaryName}

clean:
	rm -rf _output

.PHONY: all build run clean