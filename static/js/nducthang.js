var states = document.getElementById('states');
var numstate, numsymbol, initials, finals;
var nfaST = []; // bảng chuyển trạng thái cho NFA
var convert = document.getElementById('convert');
var alert = document.getElementById('alert');
var input = document.getElementById('input');
var NFA = document.getElementById('NFA');
var DFA = document.getElementById('DFA');

alert.classList.add('invisible');
convert.classList.add('invisible'); // ẩn nút convert đi

// Sự kiện khi nhấn nút ENTER
states.addEventListener('click', (event) => {
    event.preventDefault();
    alert.classList.add('invisible');
    convert.classList.add('invisible'); // ẩn nút convert đi
    numstate = document.getElementById('numstate').value;
    numsymbol = document.getElementById('numsymbol').value;
    initials = document.getElementById('initials').value.toString();
    finals = document.getElementById('finals').value.toString();

    NFA.innerHTML = "";
    DFA.innerHTML = "";
    input.innerHTML = "";
    if (!numstate || !numsymbol || !initials || !finals) {
        alert.classList.remove('invisible');
        alert.classList.add('visible');
        //alert('Vui lòng nhập đầy đủ các trường dữ liệu!');
        return;
    }
    // Khởi tạo bảng + khung nhập bảng chuyển trạng thái
    var p = document.createElement('p');
    p.innerHTML = "Nhập vào bảng chuyển đổi của NFA:";
    input.appendChild(p);
    var table = document.createElement('table');
    table.classList.add('table');
    table.classList.add('table-bordered');
    table.classList.add('table-hover');
    table.classList.add('table-sm');
    table.classList.add('table-light');
    nfaST = [];
    var tr = '<thead class="thead"><tr>';
    tr = tr + `<tr>
    <th scope="col">NFA</th>`;
    for (let i = 0; i < numsymbol; i++) {
        tr = tr + `<th scope="col">${i}</th>`;
    }
    tr = tr + '</tr> </thead>';
    tr = tr + '<tbody>';
    for (let i = 0; i < numstate; i++) {
        tr = tr + '<tr>';
        tr = tr + `<td><strong>Q${i}<strong></td>`;
        for (let j = 0; j < numsymbol; j++) {
            tr = tr + `<td><input class="form-control form-control-sm" type='text' id = stateTable${i}${j}></td>`;
        }
        tr = tr + '</tr>';
    }
    table.innerHTML = tr;
    input.appendChild(table);
    convert.classList.remove('invisible');
    convert.classList.add('visible');
});

// Sự kiện khi nhấn nút Convert!
convert.addEventListener('click', (event) => {
    event.preventDefault();
    initials_split = initials.split(',');
    finals_split = finals.split(',');
    nfaST = [];
    for (var i = 0; i < numstate; i++) {
        var rows = [];
        for (var j = 0; j < numsymbol; j++) {
            var id = "stateTable" + i.toString() + j.toString();
            rows.push(document.getElementById(id).value);
        }
        nfaST.push(rows);
    }
    displayNFA(nfaST);
    NFAtoDFA(nfaST);

});
// XỬ LÝ CHUYỂN ĐỔI NFA THÀNH DFA   
function eNFAtoDFA(NFATable) {

}
function NFAtoDFA(NFATable) {
    //console.log(NFATable);
    var nodes = [];
    var DFATable = [];
    if (NFATable) {
        nodes.push("Q0");
    }
    var index = 0;
    while (1) {
        //console.log(nodes);
        processTable(nodes[index], NFATable, DFATable, nodes);
        console.log("node chỗ này:", nodes);
        if (index == nodes.length - 1) {
            break;
        }
        else {
            index++;
        }
    }
    //console.log(DFATable);
    displayDFA(DFATable, nodes);

}
function displayDFA(a, node) {
    var x = [];
    var obj = {};
    for (let i = 0; i < node.length; i++) {
        obj = {
            id: i,
            label: `${node[i]}`
        };
        // Các trạng thái khởi tạo có viền đen nền xanh (mặc định)
        if (initials_split.indexOf(obj.label) !== -1) {
            obj.color = {
                border: 'black'
            };
        }
        // Các trạng thái kết thúc có nền vàng
        if (finals_split.indexOf(obj.label) !== -1) {
            obj.color = {
                background: 'yellow',
                border: 'red'
            };
        }
        // Các trạng thái khác có nền vàng và viền đen
        if (initials_split.indexOf(obj.label) !== -1 && finals_split.indexOf(obj.label) !== -1) {
            obj.color = {
                border: 'yellow',
            };
        }
        x.push(obj);
    }
    var nodes = new vis.DataSet(x);
    //console.log(x);
    x = [];
    for (let i = 0; i < node.length; i++) {
        for (let j = 0; j < numsymbol; j++) {
            if (a[i][j]) {

                var y = node.indexOf(a[i][j]);
                obj = {
                    from: i,
                    to: y,
                    arrows: 'to',
                    label: j.toString(),
                    color: {
                        color: 'black'
                    },
                    font: {
                        align: 'top'
                    }
                };
                var index = search(obj, x);
                if (index == -1) {
                    x.push(obj);
                } else {
                    x[index].label += "," + j.toString();
                }


            }
        }
    }

    var edges = new vis.DataSet(x);
    var data = {
        nodes: nodes,
        edges: edges
    }
    var options = {
        nodes: {
            borderWidth: 2
        },
        interaction: {
            hover: true
        },
        physics: {
            barnesHut: {
                gravitationalConstant: -4000
            }
        },
        edges: {
            arrows: {
                to: {
                    enabled: false,
                    scaleFactor: 0.5,
                    type: 'arrow'
                }
            }
        }
    };
    var container = document.getElementById('DFA');
    var network = new vis.Network(container, data, options);
}

function isExis(state, nodes) {
    // Kiểm tra xem state có tồn tại trong nodes không
    for (var i = 0; i < nodes.length; i++) {
        if (state == nodes[i]) {
            return 1; // có
        }
    }
    return 0; // không
}
function Fillter(ans) {
    if (ans == "")
        return ans;
    var temp = ans.split(",");
    var u = [];
    for (let i = 0; i < temp.length; i++) {
        if (u.indexOf(temp[i]) == -1)
            u.push(temp[i]);
    }
    u.sort();
    ans = "";
    var fflag = false;
    if (finals_split.indexOf(u[0]) !== -1) fflag = true;
    ans = u[0];
    for (let i = 1; i < u.length; i++) {
        if (finals_split.indexOf(u[i]) !== -1) fflag = true;
        ans = ans + "," + u[i];
    }
    if (fflag) finals_split.push(ans);
    return ans;

}
function processTable(state, NFATable, DFATable, nodes) {
    var splitState = state.split(',');

    var rows = [];
    for (var i = 0; i < numsymbol; i++) {
        var ans = "";
        // Hợp trạng thái
        for (var j = 0; j < splitState.length; j++) {
            var y = splitState[j].split("Q")[1];
            if (ans != "" && NFATable[y][i] != "") {
                ans = ans + "," + NFATable[y][i];
            }
            else {
                ans += NFATable[y][i];
            }
        }
        // Lọc trùng và sort lại
        ans = Fillter(ans);
        rows.push(ans);
        //console.log("row:", rows);
        // Thêm vào tập các nodes của DFA nếu nó chưa có
        if (!isExis(ans, nodes) && ans != "") {
            nodes.push(ans);
        }
    }
    //console.log(nodes);
    // Thêm hàng mới vào bảng DFA
    DFATable.push(rows);
}

// HIỂN THỊ ĐỒ THỊ CHO NFA
function displayNFA(a) {
    var x = []; // x là tập chứa các node
    var obj = {};
    for (let i = 0; i < numstate; i++) {
        obj = {
            id: i,
            label: `Q${i}`
        };
        // Các trạng thái khởi tạo có viền đen nền xanh (mặc định)
        if (initials_split.indexOf(obj.label) !== -1) {
            obj.color = {
                border: 'black'
            };
        }
        // Các trạng thái kết thúc có nền vàng
        if (finals_split.indexOf(obj.label) !== -1) {
            obj.color = {
                background: 'yellow',
                border: 'red'
            };
        }
        // Các trạng thái khác có nền vàng và viền đen
        if (initials_split.indexOf(obj.label) !== -1 && finals_split.indexOf(obj.label) !== -1) {
            obj.color = {
                border: 'yellow',
            };
        }
        x.push(obj);
    }
    var nodes = new vis.DataSet(x);
    x = []; // x là tập chứa các cạnh
    for (let i = 0; i < numstate; i++) {
        for (let j = 0; j < numsymbol; j++) {
            if (a[i][j] != "") {
                var arr = a[i][j].split(",");
                for (var k = 0; k < arr.length; k++) {
                    var y = arr[k].split('Q')[1];
                    var ss = j.toString();
                    obj = {
                        from: i,
                        to: y,
                        arrows: 'to',
                        label: ss,
                        color: {
                            color: 'black'
                        },
                        font: {
                            align: 'top'
                        }
                    };
                    var index = search(obj, x); // kiểm tra xem obj đã có trong x chưa
                    if (index == -1) {
                        x.push(obj);
                    } else {
                        x[index].label += "," + ss;
                    }
                }
            }
        }
    }

    var edges = new vis.DataSet(x);
    var data = {
        nodes: nodes,
        edges: edges
    }
    var options = {
        nodes: {
            borderWidth: 2
        },
        interaction: {
            hover: true
        },
        physics: {
            barnesHut: {
                gravitationalConstant: -4000
            }
        },
        edges: {
            arrows: {
                to: {
                    enabled: false,
                    scaleFactor: 0.5,
                    type: 'arrow'
                }
            }
        }
    };
    var container = document.getElementById('NFA');
    var network = new vis.Network(container, data, options);
}

function search(obj, x) {
    for (let i = 0; i < x.length; i++) {
        if ((obj.from == x[i].from) && (obj.to == x[i].to))
            return i;
    }
    return -1;
}
