package main

import (
	"log"

	"github.com/haormj/llama2/cmd"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	cmd.Execute()
}
