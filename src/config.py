"""
    Multimodal side-tuning for document classification
    Copyright (C) 2020  S.P. Zingaro <mailto:stefanopio.zingaro@unibo.it>.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

tobacco_img_root_dir = '../data/Tobacco3482-jpg'
tobacco_txt_root_dir = '../data/QS-OCR-small'
rlv_img_root_dir = '../data/RVL-CDIP'
rlv_txt_root_dir = '../data/QS-OCR-Large'

alphas = [
    [0.2 - 0.3 - 0.5],
    [0.2 - 0.4 - 0.4],
    [0.2 - 0.5 - 0.3],
    [0.3 - 0.2 - 0.5],
    [0.3 - 0.3 - 0.4],
    [0.3 - 0.4 - 0.3],
    [0.3 - 0.5 - 0.2],
    [0.4 - 0.2 - 0.4],
    [0.4 - 0.3 - 0.3],
    [0.4 - 0.4 - 0.2],
    [0.5 - 0.2 - 0.3],
    [0.5 - 0.3 - 0.2]
]
