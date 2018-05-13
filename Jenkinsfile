pipeline {
  agent {
    docker {
      image 'fedora:latest'
      args '--pull --rmi'
    }
    
  }
  stages {
    stage('Pull docker image') {
      steps {
        sh 'ls'
      }
    }
  }
}