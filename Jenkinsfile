stage('Package') {
    steps {
        sh '''
            echo "Packaging the project (no Docker used)"
            mkdir -p build
            cp -r src build/
            echo "Package created in ./build"
        '''
    }
}

stage('Deploy on Local Server') {
    steps {
        sh '''
            echo "Simulating deployment..."
            mkdir -p /Users/ayush/deployment
            cp -r build/* /Users/ayush/deployment/
            echo "App deployed to /Users/ayush/deployment"
        '''
    }
}
