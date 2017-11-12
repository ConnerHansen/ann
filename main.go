package main

import (
	"log"

	"github.com/connerhansen/colog"
)

// Loggers
var (
	Debug *log.Logger
	Info  *log.Logger
	Warn  *log.Logger
	Error *log.Logger
)

func main() {
	initLoggers()

	Error.Println("ANN is not currently functional. Exiting.")
}

func initLoggers() {
	colog.SetLoggingLevel(colog.LogLevelDebug)
	Debug, Info, Warn, Error = colog.SetupLoggers()
}
