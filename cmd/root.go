package cmd

import (
	"log"

	"github.com/haormj/version"
	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:     "llama2",
	Short:   "llama2 go",
	Version: version.FullVersion(),
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		log.Fatalln(err)
	}
}
