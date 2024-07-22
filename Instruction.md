This document is used to guide the specification of using GitLab to collaborate on the codes in digital.auto project.

### Basic operation
1. **Project upload/update** (e.g. Smart Wipers)：  
git clone git@gitlab.com:fsti/digital-auto.git    
git checkout smart_wipers                         
---add all your files to the directory---  
---switch to the digital.auto directory in the local---              
git add .           
git commit -m "Your message"                            
git pull  
git push --set-upstream origin smart_wipers   

### Specification:
1. **Branch definitions:**
- ***Main***: this is the master branch for the public release of the codes for digital.auto project
- ***Release***: the release branch is used for the release of version updates, and we take the name of the sub-project of digital.auto as the release branch accordingly, e.g., smart_wipers, anti-kinetosis. 
- ***Develop***: the develop branch is used for your personal code development and collaboration with others, and the responsible person distinguishes his or her name as a suffix for the sub-project, e.g., smart_wipers-chris, anti-kinetosis-chris.
 
#### Hints:
1. Everyone must create a **.md** file as logs in the develop branch to record your changes.
