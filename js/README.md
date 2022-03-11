# hydra

> Hydra Graphs

## Spec

### Desired Features

- OHLC chart
- Change OHLC sampling/resolution by zoom
- Draw markers on top of chart
- Sliders & buttons and things to select datasets and subsets
- hoverinfotooltips
- hover a point to show the waveform, click to persist

###

- Server loads all data from parquets into memory
-     Compare memory usage of array of objects vs danfo.js
- Client retrieves desired data
-     Either
-         retrieve all data within visible time range
-         Retrieve preprocessed data within time range
-         Retrieve subset of preprocessed data within time range based on other filters
- Client renders data
-     Reacts to sliding by loading/streaming more data
-     Reacts to zoom by loading resampled data
-     reacts to form changes by loading filtered data
-     Cache data in browser so we don't have to reload the same data

## About

This project uses [Feathers](http://feathersjs.com). An open source web framework for building modern real-time applications.

## Getting Started

Getting up and running is as easy as 1, 2, 3.

1. Make sure you have [NodeJS](https://nodejs.org/) and [npm](https://www.npmjs.com/) installed.
2. Install your dependencies

   ```
   cd path/to/hydra
   npm install
   ```

3. Start your app

   ```
   npm start
   ```

## Testing

Simply run `npm test` and all your tests in the `test/` directory will be run.

## Scaffolding

Feathers has a powerful command line interface. Here are a few things it can do:

```
$ npm install -g @feathersjs/cli          # Install Feathers CLI

$ feathers generate service               # Generate a new Service
$ feathers generate hook                  # Generate a new Hook
$ feathers help                           # Show all commands
```

## Help

For more information on all the things you can do with Feathers visit [docs.feathersjs.com](http://docs.feathersjs.com).
