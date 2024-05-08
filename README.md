> Author: [Erik Buis](https://github.com/ErikBuis) \
> Email: [ebbuis@gmail.com](mailto:ebbuis@gmail.com) \
> Date of creation of this file: 8 March, 2024 \
> Copyright Erik Buis (2024). See LICENCE.md for details.


# Utility functions
TODO: Add project description.


## Project Structure
The repository is organized as follows:
<pre>
.
├── ℹ️ <b>README.md</b>: This file.
├── ℹ️ <b>LICENCE.md</b>: Licence file.
├── 📁 <b>modules</b>: General utility functions organized by the Python module they correlate most with. Each file is completely independent from the rest of the files, so you can safely copy-paste a single file into your project.
├── 📁 <b>cheatsheet</b>: Functions to be used for competitive programming problems.
├── 📁 <b>batched</b>: General utility functions like in the 'modules' directory, but here they are batched using torch.Tensor operations.
└── 📁 <b>custom</b>: Custom utility modules.
</pre>


## Testing
The `tests` directory contaisn tests for all modules. The tests are written using the unittest module from the Python standard library.

To run the tests, run the following command from the root directory of the repository:
```bash
python3 -m pytest tests/**/*.py
```
You can also run the tests using `python3 -m unittest tests/**/*.py`, but pytest is recommended because it provides more detailed and coloured output. You may need to install pytest first using `python3 -m pip install pytest`.


## Copyright
THIS REPOSITORY IS LICENCED UNDER THE APACHE LICENCE 2.0.

FOR TERMS AND CONDITIONS, SEE THE LICENCE FILE OR VISIT:
http://www.apache.org/licenses/LICENSE-2.0

The code is sourced from the repository located at:
https://github.com/ErikBuis/utils

The copyright belongs to Erik Buis (2024).

Commercial use, modification, distribution, and private use are allowed under
the condition that the original author and licence are mentioned in the source
code. See LICENCE.md for details.
