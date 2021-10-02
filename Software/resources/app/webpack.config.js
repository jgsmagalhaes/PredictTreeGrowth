const ExtractTextPlugin = require('extract-text-webpack-plugin');
const SpawnPlugin = require('webpack-spawn-plugin');

module.exports = {

    watch: true,

    target: 'electron-renderer',

    entry: './app/src/renderer_process.js',

    output: {
        path: __dirname + '/app/build',
        publicPath: 'build/',
        filename: 'bundle.js'
    },

    module: {
        rules: [
            {
                test: /\.js?$/,
                loader: 'babel-loader',
                options: {
                    presets: ['react'],
										plugins: [
									    "babel-plugin-transform-object-rest-spread",
											"transform-class-properties"
									  ],
                }
            },
            {
                test: /\.css$/,
                loader: ExtractTextPlugin.extract({
                  loader: 'css-loader',
                  options: {
                    modules: true
                  }
                })
            },
            {
                test: /\.(png|jpg|gif|svg)$/,
                loader: 'file-loader',
                query: {
                    name: '[name].[ext]?[hash]'
                }
            }
        ]
    },

    plugins: [
        new ExtractTextPlugin({
            filename: 'bundle.css',
            disable: false,
            allChunks: true
        })
    ],

    resolve: {
      extensions: ['.js', '.json', '.jsx']
    }

}
