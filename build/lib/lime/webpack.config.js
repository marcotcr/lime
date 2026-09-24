var path = require('path');
var webpack = require('webpack');

module.exports = {
    entry: './js/main.js',
    output: {
        path: __dirname,
        filename: 'bundle.js',
        library: 'lime'
    },
    module: {
        loaders: [
            {
                loader: 'babel-loader',
                test: path.join(__dirname, 'js'),
                query: {
                   presets: 'es2015-ie',
                  },

            },
            {
              test: /\.css$/,
              loaders: ['style-loader', 'css-loader'],

            }

        ]
    },
    plugins: [
        // Avoid publishing files when compilation fails
        new webpack.NoErrorsPlugin()
    ],
    stats: {
        // Nice colored output
        colors: true
    },
    // Create Sourcemaps for the bundle
    devtool: 'source-map',
};

